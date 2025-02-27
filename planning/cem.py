import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device
from .objectives import chamfer_distance as chamf

class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None, batch=False):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        if batch:
            trans_obs_g = [ move_to_device(self.preprocessor.transform_obs(g), self.device) for g in obs_g ]
            z_obs_g = [self.wm.encode_obs(g) for g in trans_obs_g]
        else:
            trans_obs_g = move_to_device(
                self.preprocessor.transform_obs(obs_g), self.device
            )
            z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for idx in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                if batch:
                    cur_z_obs_g = [{}]*len(z_obs_g)
                    for i in range(len(cur_z_obs_g)):
                        for key in z_obs_g[i]:
                            if key=="reach":
                                cur_z_obs_g[i]["reach"] = z_obs_g[i]["reach"]
                            else:
                                cur_z_obs_g[i][key] = repeat(
                                    z_obs_g[i][key][traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                                )
                else:
                    cur_z_obs_g = {
                        key: repeat(
                            arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )
                        for key, arr in z_obs_g.items()
                    }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = 0
                #loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                #print(loss.shape)
                #print('loss is \n\n\n\n')
                cdavoid = {}
                if not batch:
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                else:
                    '''reach_loss = []
                    avoid_loss = []
                    for czog in range(len(cur_z_obs_g)):
                        l = self.objective_fn(i_z_obses, cur_z_obs_g[czog])
                        if cur_z_obs_g[czog]["reach"]:
                            reach_loss.append(l)
                        else:
                            avoid_loss.append(l)
                            # print out chamfer distance between izobses and current avoid
                            cd = chamf(i_z_obses, cur_z_obs_g[czog])
                            cdavoid.update({f"CD to avoid {czog}": cd})
                            print(f"\tCD to avoid {czog}: {cd}")
                    if reach_loss == []: loss = np.max(avoid_loss)
                    elif avoid_loss == []: loss = np.min(reach_loss)
                    else: loss = np.min(reach_loss) + np.max(avoid_loss)'''
                    for czog in range(len(cur_z_obs_g)):
                        l = self.objective_fn(i_z_obses, cur_z_obs_g[czog])
                        if cur_z_obs_g[czog]["reach"]:
                            loss = loss + l
                            cd = chamf(i_z_obses["visual"], cur_z_obs_g[czog]["visual"])
                            cdavoid.update({f"CD to reach {czog}": cd})
                            print(f"\tCD to reach {czog}: {cd}")
                        else:
                            loss = loss - l
                            cd = chamf(i_z_obses, cur_z_obs_g[czog])
                            cdavoid.update({f"CD to avoid {czog}": cd})
                            print(f"\tCD to avoid {czog}: {cd}")
                
                   
                topk_idx = torch.argsort(loss)[: self.topk]
                #topk_idx = torch.argsort(loss)[0][:, self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": idx + 1}
            )
            if self.evaluator is not None and idx % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{idx+1}", batch=batch
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": idx + 1})
                logs.update(cdavoid)
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid
