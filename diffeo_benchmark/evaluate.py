import os
import argparse
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from diffeo_benchmark.diffeo.generator import DiffeoConfig, DiffeoGenerator
from diffeo_benchmark.data.dataset import DiffeoEpisodeDataset, collate_episodes
from diffeo_benchmark.models.transformer import DiffeoTransformer
from diffeo_benchmark.models.deepsets import DeepSetsMLP
from diffeo_benchmark.models.invertible_flow import InvertibleFlowTransformer
from diffeo_benchmark.metrics.evaluator import (
    compute_coordinate_rmse,
    compute_relative_coord_error,
    evaluate_jacobian_error,
    evaluate_inverse_reconstruction
)

def load_model(model_name: str, checkpoint_path: str, dim: int = 8, device: torch.device = torch.device("cpu")):
    if model_name == "transformer":
        model = DiffeoTransformer(dim=dim, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256)
    elif model_name == "deepsets":
        model = DeepSetsMLP(dim=dim, hidden_dim=256, latent_dim=256, num_enc_layers=4, num_dec_layers=4)
    elif model_name == "invertible_flow":
        model = InvertibleFlowTransformer(dim=dim, cond_dim=128, num_enc_layers=3, num_flow_layers=4, hidden_dim=128)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with random weights.")
        
    model.to(device)
    model.eval()
    return model

class OracleModel:
    """Ground truth oracle model for evaluation calibration"""
    def __init__(self, dataset: DiffeoEpisodeDataset):
        self.dataset = dataset

    def __call__(self, x_ctx, y_ctx, x_qry, episode_idx=0):
        # Oracle directly transforms query points using true diffeo
        phi = self.dataset.get_diffeo(episode_idx)
        return phi.forward(x_qry)

def evaluate_single_setting(
    model,
    dataset: DiffeoEpisodeDataset,
    device: torch.device,
    is_oracle: bool = False,
    num_eval_episodes: int = 100,
    eval_jacobian: bool = True,
    eval_inverse: bool = True
):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_episodes)
    
    total_rmse = []
    total_rel = []
    jac_errors = []
    pos_det_rates = []
    inv_errors = []
    
    episodes_evaluated = 0
    t0 = time.time()
    
    for b_idx, batch in enumerate(loader):
        if episodes_evaluated >= num_eval_episodes:
            break
            
        x_ctx = batch["x_ctx"].to(device)
        y_ctx = batch["y_ctx"].to(device)
        x_qry = batch["x_qry"].to(device)
        y_qry = batch["y_qry"].to(device)
        bs = x_ctx.size(0)
        
        if is_oracle:
            preds = []
            for i in range(bs):
                ep_idx = batch["episode_idx"][i].item()
                phi = dataset.get_diffeo(ep_idx).to(device)
                preds.append(phi.forward(x_qry[i:i+1]))
            y_pred = torch.cat(preds, dim=0)
        else:
            with torch.no_grad():
                y_pred = model(x_ctx, y_ctx, x_qry)
                
        rmse = compute_coordinate_rmse(y_pred, y_qry)
        rel_err = compute_relative_coord_error(y_pred, y_qry)
        total_rmse.append(rmse)
        total_rel.append(rel_err)
        
        # Detailed geometry evaluation on first few episodes
        if eval_jacobian and episodes_evaluated < min(10, num_eval_episodes):
            for i in range(min(2, bs)):
                ep_idx = batch["episode_idx"][i].item()
                phi = dataset.get_diffeo(ep_idx).to(device)
                
                if is_oracle:
                    jac_errors.append(0.0)
                    pos_det_rates.append(1.0)
                else:
                    jac_res = evaluate_jacobian_error(
                        model=model,
                        x_ctx=x_ctx[i:i+1],
                        y_ctx=y_ctx[i:i+1],
                        x_qry=x_qry[i:i+1],
                        true_diffeo=phi,
                        num_eval_points=8
                    )
                    jac_errors.append(jac_res["jac_rel_error"])
                    pos_det_rates.append(jac_res["positive_det_rate"])
                    
                if eval_inverse and not is_oracle and len(inv_errors) < 5:
                    inv_err = evaluate_inverse_reconstruction(
                        model=model,
                        x_ctx=x_ctx[i:i+1],
                        y_ctx=y_ctx[i:i+1],
                        y_qry=y_pred[i:i+1],
                        x_qry=x_qry[i:i+1],
                        num_steps=60
                    )
                    inv_errors.append(inv_err)
                    
        episodes_evaluated += bs
        
    elapsed = time.time() - t0
    
    return {
        "coord_rmse": float(np.mean(total_rmse)),
        "coord_rel_err": float(np.mean(total_rel)),
        "jac_rel_err": float(np.mean(jac_errors)) if jac_errors else 0.0,
        "positive_det_rate": float(np.mean(pos_det_rates)) if pos_det_rates else 1.0,
        "inverse_rmse": float(np.mean(inv_errors)) if inv_errors else 0.0,
        "eval_time_sec": elapsed
    }
