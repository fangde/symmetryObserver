import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np

def compute_coordinate_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Coordinate RMSE across batch and points.
    """
    diff = y_pred - y_true
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    return rmse

def compute_relative_coord_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    ||Y_pred - Y_true||_F / ||Y_true||_F
    """
    diff = y_pred - y_true
    rel_err = (torch.norm(diff) / (torch.norm(y_true) + 1e-8)).item()
    return rel_err

def compute_model_jacobian(
    model: nn.Module,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_qry_single: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes \hat{J} = d \hat{Y} / d X for a single query point x_qry_single: (D,)
    under context x_ctx: (1, K, D), y_ctx: (1, K, D).
    Returns: (D, D) tensor.
    """
    def forward_fn(x_point):
        # x_point: (D,) -> (1, 1, D)
        pred = model(x_ctx, y_ctx, x_point.unsqueeze(0).unsqueeze(0))
        return pred.squeeze(0).squeeze(0) # (D,)
        
    J_hat = torch.autograd.functional.jacobian(forward_fn, x_qry_single)
    return J_hat

def evaluate_jacobian_error(
    model: nn.Module,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_qry: torch.Tensor,
    true_diffeo,
    num_eval_points: int = 16
) -> Dict[str, float]:
    """
    Evaluates Jacobian relative Frobenius error and local orientation preservation (det(J) > 0).
    Evaluated on a subset of query points (e.g. 16 per episode) to keep compute efficient.
    """
    model.eval()
    device = x_qry.device
    dim = x_qry.shape[-1]
    
    # We take the first episode in the batch
    ctx_x = x_ctx[0:1] # (1, K, D)
    ctx_y = y_ctx[0:1] # (1, K, D)
    qry_x = x_qry[0, :num_eval_points] # (num_eval, D)
    
    # Ground truth analytic Jacobians: (num_eval, D, D)
    J_true = true_diffeo.jacobian(qry_x) # (num_eval, D, D)
    
    rel_errors = []
    positive_det_count = 0
    
    for i in range(num_eval_points):
        xi = qry_x[i]
        Ji_true = J_true[i]
        
        Ji_model = compute_model_jacobian(model, ctx_x, ctx_y, xi)
        
        diff = torch.norm(Ji_model - Ji_true, p="fro")
        scale = torch.norm(Ji_true, p="fro") + 1e-8
        rel_errors.append((diff / scale).item())
        
        det_val = torch.linalg.det(Ji_model).item()
        if det_val > 0:
            positive_det_count += 1
            
    mean_jac_error = float(np.mean(rel_errors))
    pos_det_rate = positive_det_count / num_eval_points
    
    return {
        "jac_rel_error": mean_jac_error,
        "positive_det_rate": pos_det_rate
    }

def evaluate_inverse_reconstruction(
    model: nn.Module,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    y_qry: torch.Tensor,
    x_qry: torch.Tensor,
    num_steps: int = 100,
    lr: float = 0.1
) -> float:
    """
    Evaluates inverse error:
    If model has explicit .inverse (Model C), calls it directly.
    Otherwise, uses gradient-based inversion to find x* = argmin_x ||F(x) - y||^2.
    """
    model.eval()
    if hasattr(model, "inverse") and callable(getattr(model, "inverse")):
        with torch.no_grad():
            x_rec = model.inverse(x_ctx, y_ctx, y_qry)
            err = torch.sqrt(torch.mean((x_rec - x_qry) ** 2)).item()
            return err
            
    # For models without explicit inverse, test optimization-based inversion
    # on a small batch of query points (e.g., 16 points)
    B = min(2, x_ctx.shape[0])
    M = min(16, y_qry.shape[1])
    
    sub_x_ctx = x_ctx[:B]
    sub_y_ctx = y_ctx[:B]
    target_y = y_qry[:B, :M].detach()
    true_x = x_qry[:B, :M]
    
    # Initialize optimization at target_y
    x_opt = target_y.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_opt], lr=lr)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        y_cand = model(sub_x_ctx, sub_y_ctx, x_opt)
        loss = torch.mean((y_cand - target_y) ** 2)
        loss.backward()
        optimizer.step()
        
    inv_err = torch.sqrt(torch.mean((x_opt.detach() - true_x) ** 2)).item()
    return inv_err
