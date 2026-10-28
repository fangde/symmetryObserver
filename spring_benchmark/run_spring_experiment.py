import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from spring_benchmark.physics.spring_system import DuffingOscillator, CoupledSpringChain8D
from spring_benchmark.data import SpringEpisodeDataset
from diffeo_benchmark.data.dataset import collate_episodes
from diffeo_benchmark.models.transformer import DiffeoTransformer
from diffeo_benchmark.metrics.evaluator import compute_coordinate_rmse, compute_relative_coord_error

def train_spring_transformer(is_nonlinear: bool, epochs: int = 5, episodes: int = 1200, dim: int = 8, device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    tag = "nonlinear" if is_nonlinear else "linear"
    print(f"\n>>> Training Transformer on {tag.upper()} Spring (Dim={dim}, Episodes={episodes}, Epochs={epochs})...")
    
    train_ds = SpringEpisodeDataset(num_episodes=episodes, dim=dim, is_nonlinear=is_nonlinear, num_context=32, num_query=128, seed=42)
    val_ds = SpringEpisodeDataset(num_episodes=200, dim=dim, is_nonlinear=is_nonlinear, num_context=32, num_query=128, seed=999)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_episodes)
    
    model = DiffeoTransformer(dim=dim, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_rmse = float("inf")
    save_path = f"spring_benchmark/results/transformer_{tag}_d{dim}.pt"
    
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            x_ctx = batch["x_ctx"].to(device)
            y_ctx = batch["y_ctx"].to(device)
            x_qry = batch["x_qry"].to(device)
            y_qry = batch["y_qry"].to(device)
            
            optimizer.zero_grad()
            pred = model(x_ctx, y_ctx, x_qry)
            loss = criterion(pred, y_qry)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x_ctx.size(0)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_rmse_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x_ctx = batch["x_ctx"].to(device)
                y_ctx = batch["y_ctx"].to(device)
                x_qry = batch["x_qry"].to(device)
                y_qry = batch["y_qry"].to(device)
                pred = model(x_ctx, y_ctx, x_qry)
                rmse = compute_coordinate_rmse(pred, y_qry)
                val_rmse_sum += rmse * x_ctx.size(0)
                val_count += x_ctx.size(0)
        val_rmse = val_rmse_sum / val_count
        print(f"[{tag.upper()}] Epoch {ep:02d}/{epochs:02d} | Train MSE: {train_loss:.5f} | Val RMSE: {val_rmse:.4f} | Time: {time.time()-t0:.1f}s")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            
    print(f"[{tag.upper()}] Best Val RMSE: {best_rmse:.4f} -> Checkpoint saved.")
    return save_path

def evaluate_spring_model(model_path: str, is_nonlinear: bool, dim: int = 8, num_context: int = 32):
    device = torch.device("cpu")
    model = DiffeoTransformer(dim=dim, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_ds = SpringEpisodeDataset(num_episodes=50, dim=dim, is_nonlinear=is_nonlinear, num_context=num_context, num_query=128, seed=12345)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_episodes)
    
    rmses = []
    rel_errs = []
    energy_errors = []
    
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x_ctx = batch["x_ctx"]
            y_ctx = batch["y_ctx"]
            x_qry = batch["x_qry"]
            y_qry = batch["y_qry"]
            
            pred = model(x_ctx, y_ctx, x_qry)
            rmse = compute_coordinate_rmse(pred, y_qry)
            rel = compute_relative_coord_error(pred, y_qry)
            rmses.append(rmse)
            rel_errs.append(rel)
            
            # Compute energy error using physical Hamiltonian
            for i in range(x_ctx.size(0)):
                ep_idx = batch["episode_idx"][i].item()
                sys = test_ds.get_system(ep_idx)
                
                # Energy of predicted states vs true states
                E_pred = sys.energy(pred[i])
                E_true = sys.energy(y_qry[i])
                e_rel = torch.mean(torch.abs(E_pred - E_true) / (torch.abs(E_true) + 1e-6)).item()
                energy_errors.append(e_rel)
                
    return {
        "rmse": float(np.mean(rmses)),
        "rel_err": float(np.mean(rel_errs)),
        "energy_rel_error": float(np.mean(energy_errors))
    }

def main():
    os.makedirs("spring_benchmark/results", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dim = 8 # 4-DOF Coupled Spring Chain (8D: 4 positions, 4 momenta)
    
    # 1. Train linear spring model
    linear_ckpt = train_spring_transformer(is_nonlinear=False, epochs=5, episodes=1200, dim=dim, device=device)
    # 2. Train nonlinear spring model
    nonlinear_ckpt = train_spring_transformer(is_nonlinear=True, epochs=5, episodes=1200, dim=dim, device=device)
    
    # 3. Evaluate both
    print("\n" + "="*70)
    print("           PHYSICAL BENCHMARK: LINEAR vs. NONLINEAR SPRING          ")
    print("="*70)
    
    res_linear = evaluate_spring_model(linear_ckpt, is_nonlinear=False, dim=dim, num_context=32)
    res_nonlinear = evaluate_spring_model(nonlinear_ckpt, is_nonlinear=True, dim=dim, num_context=32)
    
    # Also evaluate context scaling
    res_linear_k128 = evaluate_spring_model(linear_ckpt, is_nonlinear=False, dim=dim, num_context=128)
    res_nonlinear_k128 = evaluate_spring_model(nonlinear_ckpt, is_nonlinear=True, dim=dim, num_context=128)
    
    summary = {
        "Linear Spring (K=32)": res_linear,
        "Linear Spring (K=128)": res_linear_k128,
        "Nonlinear Spring (K=32)": res_nonlinear,
        "Nonlinear Spring (K=128)": res_nonlinear_k128
    }
    
    with open("spring_benchmark/results/spring_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"| {'System Configuration':<26} | {'Coord RMSE':<12} | {'Rel Err (%)':<14} | {'Energy Rel Err (%)':<20} |")
    print(f"|{'-'*28}|{'-'*14}|{'-'*16}|{'-'*22}|")
    for name, r in summary.items():
        print(f"| {name:<26} | {r['rmse']:<12.4f} | {r['rel_err']*100:<14.2f}% | {r['energy_rel_error']*100:<20.2f}% |")
    print("="*70)
    print("Results saved to spring_benchmark/results/spring_experiment_summary.json")

if __name__ == "__main__":
    main()
