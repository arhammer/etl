import numpy as np
import torch
import torch.nn as nn
import scipy.spatial.distance as dist

def chamfer_distance(x, y):
    x_flat = x.reshape(x.shape[0], x.shape[1], -1)
    y_flat = y.reshape(y.shape[0], y.shape[1], -1)
    dist_matrix = torch.cdist(x_flat, y_flat, p=2)
    min_dist_x = dist_matrix.min(dim=2)[0]
    min_dist_y = dist_matrix.min(dim=1)[0]
    return min_dist_x.sum(dim=1).sum() + min_dist_y.sum(dim=1).sum()

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
        #print(f"pred visual shape: {z_obs_pred['visual'].shape}")
        #print(f"pred proprio shape: {z_obs_pred['proprio'].shape}")
        #print(f"tgt visual shape: {z_obs_tgt['visual'].shape}")
        #print(f"tgt proprio shape: {z_obs_tgt['proprio'].shape}")
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
        #print(f"visual dim: {tuple(range(1, z_obs_pred['visual'].ndim))}")
        #print(f"loss proprio shape: {loss_visual.shape}")
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        #print(f"proprio dim: {tuple(range(1, z_obs_pred['proprio'].ndim))}")
        #print(f"loss proprio shape: {loss_proprio.shape}")
        loss = loss_visual + alpha * loss_proprio
        #print(f"loss shape: {loss.shape}")
        return loss

    def objective_fn_mse_all(z_obs_pred, z_obs_tgt, batch=False):
        if batch:
            z_obs_tgt = z_obs_tgt[0]
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
        cost = 1 - nn.functional.cosine_similarity(z_obs_pred["visual"][:,-1:].reshape(300,-1), z_obs_tgt["visual"].reshape(300,-1)) # higher similarity is lower cost
        return cost
    
    def objective_fn_mahal(z_obs_pred, z_obs_tgt):
        e1 = z_obs_pred["visual"][:,-1:]
        e2 = z_obs_tgt["visual"]
        # compute covariance matrix inverse
        inv_cov = torch.linalg.inv(torch.cov([e1, e2], rowvar=False))
        cost = 1 - torch.mahalanobis(e1, e2, inv_cov)
        return cost

    def objective_fn_cd(z_obs_pred, z_obs_tgt):
        return chamfer_distance(z_obs_pred["visual"][:,-1:], z_obs_tgt["visual"])

    def objective_fn_l1(z_obs_pred, z_obs_tgt):
        cost =  1 - (torch.abs(z_obs_pred["visual"] - z_obs_tgt["visual"])).sum(dim=tuple(range(1, z_obs_pred["visual"].ndim)))
        return cost

    if mode == "mse_last":
        return objective_fn_mse_last
    elif mode == "mse_all":
        return objective_fn_mse_all
    elif mode == "cos":
        return objective_fn_cos
    elif mode == "mahal":
        return objective_fn_mahal
    elif mode == "l1":
        return objective_fn_l1
    elif mode == "cd":
        return objective_fn_cd
    else:
        raise NotImplementedError
