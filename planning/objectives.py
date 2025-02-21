import numpy as np
import torch
import torch.nn as nn
import scipy.spatial.distance as dist

# this is a direct copy/past from ../../env/deformable_env/FlexEnvWrapper
# i could not import this and instead of fighting with it I'm just doing this
def chamfer_distance(x, y):
    # x: [B, N, D]
    # y: [B, M, D]
    # NOTE: only the first 3 dim is taken!
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
    dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
    dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
    dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M
    return dis_xy + dis_yx

def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_mse_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_mse_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss
    
    def objective_fn_cos(z_obs_pred, z_obs_tgt):
        cost = 1 - dist.cosine(z_obs_pred["visual"][:,-1:], z_obs_tgt["visual"]) # higher similarity is lower cost
        return cost 
    
    def objective_fn_mahal(z_obs_pred, z_obs_tgt):
        e1 = z_obs_pred["visual"][:,-1:]
        e2 = z_obs_tgt["visual"]
        # compute covariance matrix inverse
        inv_cov = np.linalg.inv(np.cov([e1, e2], rowvar=False))
        cost = 1 - dist.mahalanobis(e1, e2, inv_cov)
        return cost 

    def objective_fn_cd(z_obs_pred, z_obs_tgt):
        return chamfer_distance(z_obs_pred["visual"][:,-1:], z_obs_tgt["visual"])

    if mode == "mse_last":
        return objective_fn_last
    elif mode == "mse_all":
        return objective_fn_all
    elif mode == "cos":
        return objective_fn_cos
    elif mode == "mahal":
        return objective_fn_mahal
    else:
        raise NotImplementedError
