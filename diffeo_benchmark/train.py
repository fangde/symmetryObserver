import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffeo_benchmark.diffeo.generator import DiffeoConfig
from diffeo_benchmark.data.dataset import DiffeoEpisodeDataset, collate_episodes
from diffeo_benchmark.models.transformer import DiffeoTransformer
from diffeo_benchmark.models.deepsets import DeepSetsMLP
from diffeo_benchmark.models.invertible_flow import InvertibleFlowTransformer
from diffeo_benchmark.metrics.evaluator import compute_coordinate_rmse, compute_relative_coord_error

def get_model(model_name: str, dim: int = 8):
    if model_name == "transformer":
        return DiffeoTransformer(
            dim=dim,
            d_model=128,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            residual_output=True
        )
    elif model_name == "deepsets":
        return DeepSetsMLP(
            dim=dim,
            hidden_dim=256,
            latent_dim=256,
            num_enc_layers=4,
            num_dec_layers=4,
            residual_output=True
        )
    elif model_name == "invertible_flow":
        return InvertibleFlowTransformer(
            dim=dim,
            cond_dim=128,
            num_enc_layers=3,
            num_flow_layers=4,
            hidden_dim=128
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train():
    parser = argparse.ArgumentParser(description="Train Diffeomorphic Transformation Benchmark Models")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "deepsets", "invertible_flow"])
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_context", type=int, default=32)
    parser.add_argument("--num_query", type=int, default=128)
    parser.add_argument("--num_train_episodes", type=int, default=10000)
    parser.add_argument("--num_val_episodes", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device} | Model: {args.model} | Dim: {args.dim} | Layers: {args.num_layers} | Context: {args.num_context}")

    torch.manual_seed(args.seed)

    # Configs
    train_config = DiffeoConfig(dim=args.dim, num_layers=args.num_layers, s_std=0.3, a_std=0.4, w_std=0.5, b_std=0.5)
    val_config = DiffeoConfig(dim=args.dim, num_layers=args.num_layers, s_std=0.3, a_std=0.4, w_std=0.5, b_std=0.5)

    print("Building datasets...")
    train_dataset = DiffeoEpisodeDataset(
        num_episodes=args.num_train_episodes,
        config=train_config,
        num_context=args.num_context,
        num_query=args.num_query,
        seed=args.seed,
        cache_diffeos=True
    )
    val_dataset = DiffeoEpisodeDataset(
        num_episodes=args.num_val_episodes,
        config=val_config,
        num_context=args.num_context,
        num_query=args.num_query,
        seed=args.seed + 9999,
        cache_diffeos=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_episodes)

    model = get_model(args.model, dim=args.dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_rmse = float("inf")
    save_path = os.path.join(args.save_dir, f"{args.model}_l{args.num_layers}_k{args.num_context}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            x_ctx = batch["x_ctx"].to(device)
            y_ctx = batch["y_ctx"].to(device)
            x_qry = batch["x_qry"].to(device)
            y_qry = batch["y_qry"].to(device)

            optimizer.zero_grad()
            y_pred = model(x_ctx, y_ctx, x_qry)
            loss = criterion(y_pred, y_qry)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * x_ctx.size(0)

        train_loss /= len(train_dataset)
        elapsed = time.time() - start_time

        # Validation
        model.eval()
        val_rmse_sum = 0.0
        val_rel_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x_ctx = batch["x_ctx"].to(device)
                y_ctx = batch["y_ctx"].to(device)
                x_qry = batch["x_qry"].to(device)
                y_qry = batch["y_qry"].to(device)

                y_pred = model(x_ctx, y_ctx, x_qry)
                rmse = compute_coordinate_rmse(y_pred, y_qry)
                rel_err = compute_relative_coord_error(y_pred, y_qry)
                
                bs = x_ctx.size(0)
                val_rmse_sum += rmse * bs
                val_rel_sum += rel_err * bs
                val_samples += bs

        val_rmse = val_rmse_sum / val_samples
        val_rel = val_rel_sum / val_samples

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train MSE: {train_loss:.5f} | Val RMSE: {val_rmse:.4f} | Val RelErr: {val_rel*100:.2f}% | Time: {elapsed:.1f}s")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_rmse": val_rmse,
                "val_rel": val_rel,
                "config": vars(args)
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val RMSE: {best_val_rmse:.4f})")

    print(f"\nTraining completed! Best Val RMSE: {best_val_rmse:.4f}")

if __name__ == "__main__":
    train()
