import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spring_benchmark.physics.spring_system import DuffingOscillator
from diffeo_benchmark.models.transformer import DiffeoTransformer

def train_2d_transformer(is_nonlinear: bool, epochs: int = 6, episodes: int = 1500, dt: float = 0.6):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dim = 2
    tag = "nonlinear" if is_nonlinear else "linear"
    save_path = f"spring_benchmark/results/transformer_2d_{tag}.pt"
    
    # Fast in-memory training
    rng = np.random.RandomState(42)
    k_vals = rng.uniform(0.8, 2.5, size=episodes)
    if is_nonlinear:
        beta_vals = rng.uniform(0.8, 2.5, size=episodes)
    else:
        beta_vals = np.zeros(episodes)
        
    all_x_ctx, all_y_ctx, all_x_qry, all_y_qry = [], [], [], []
    for ep in range(episodes):
        osc = DuffingOscillator(k=float(k_vals[ep]), beta=float(beta_vals[ep]))
        t_gen = torch.Generator().manual_seed(ep * 997 + 13)
        x_c = torch.randn(32, dim, generator=t_gen) * 1.3
        y_c = osc.flow_map(x_c, dt=dt)
        x_q = torch.randn(64, dim, generator=t_gen) * 1.3
        y_q = osc.flow_map(x_q, dt=dt)
        all_x_ctx.append(x_c)
        all_y_ctx.append(y_c)
        all_x_qry.append(x_q)
        all_y_qry.append(y_q)
        
    X_ctx = torch.stack(all_x_ctx) # (B, 32, 2)
    Y_ctx = torch.stack(all_y_ctx)
    X_qry = torch.stack(all_x_qry) # (B, 64, 2)
    Y_qry = torch.stack(all_y_qry)
    
    model = DiffeoTransformer(dim=2, d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    batch_size = 32
    num_batches = episodes // batch_size
    
    print(f"Training 2D Transformer ({tag})...")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(episodes)
        total_loss = 0.0
        for b in range(num_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            xc = X_ctx[idx].to(device)
            yc = Y_ctx[idx].to(device)
            xq = X_qry[idx].to(device)
            yq = Y_qry[idx].to(device)
            
            optimizer.zero_grad()
            pred = model(xc, yc, xq)
            loss = criterion(pred, yq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")
    return save_path

def plot_10_correspondences():
    os.makedirs("spring_benchmark/results", exist_ok=True)
    dt = 0.6
    
    # Train 2D models if not exist
    linear_ckpt = "spring_benchmark/results/transformer_2d_linear.pt"
    nonlinear_ckpt = "spring_benchmark/results/transformer_2d_nonlinear.pt"
    if not os.path.exists(linear_ckpt):
        linear_ckpt = train_2d_transformer(is_nonlinear=False, dt=dt)
    if not os.path.exists(nonlinear_ckpt):
        nonlinear_ckpt = train_2d_transformer(is_nonlinear=True, dt=dt)
        
    # Setup test oscillators
    test_k = 1.8
    test_beta_linear = 0.0
    test_beta_nonlinear = 1.6
    
    osc_linear = DuffingOscillator(k=test_k, beta=test_beta_linear)
    osc_nonlinear = DuffingOscillator(k=test_k, beta=test_beta_nonlinear)
    
    # Load models
    model_lin = DiffeoTransformer(dim=2, d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=192)
    model_lin.load_state_dict(torch.load(linear_ckpt, map_location="cpu"))
    model_lin.eval()
    
    model_non = DiffeoTransformer(dim=2, d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=192)
    model_non.load_state_dict(torch.load(nonlinear_ckpt, map_location="cpu"))
    model_non.eval()
    
    # Context points (K=32)
    rng_ctx = torch.Generator().manual_seed(101)
    x_ctx_lin = torch.randn(1, 32, 2, generator=rng_ctx) * 1.2
    y_ctx_lin = osc_linear.flow_map(x_ctx_lin, dt=dt)
    
    rng_ctx2 = torch.Generator().manual_seed(202)
    x_ctx_non = torch.randn(1, 32, 2, generator=rng_ctx2) * 1.2
    y_ctx_non = osc_nonlinear.flow_map(x_ctx_non, dt=dt)
    
    # 10 representative query initial points arranged nicely across phase space
    # (different radii and angles)
    angles = np.linspace(0.1, 1.9 * np.pi, 10)
    radii = np.linspace(0.6, 2.2, 10)
    q0 = radii * np.cos(angles)
    p0 = radii * np.sin(angles)
    x_query = torch.tensor(np.stack([q0, p0], axis=-1), dtype=torch.float32).unsqueeze(0) # (1, 10, 2)
    
    # Ground truth targets
    y_true_lin = osc_linear.flow_map(x_query, dt=dt).squeeze(0).numpy() # (10, 2)
    y_true_non = osc_nonlinear.flow_map(x_query, dt=dt).squeeze(0).numpy() # (10, 2)
    
    # Model predictions
    with torch.no_grad():
        y_pred_lin = model_lin(x_ctx_lin, y_ctx_lin, x_query).squeeze(0).numpy()
        y_pred_non = model_non(x_ctx_non, y_ctx_non, x_query).squeeze(0).numpy()
        
    x_q_np = x_query.squeeze(0).numpy() # (10, 2)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    
    systems = [
        ("Linear Spring (k=1.8, beta=0.0)", osc_linear, x_ctx_lin, y_ctx_lin, y_true_lin, y_pred_lin, axes[0]),
        ("Nonlinear Spring (k=1.8, beta=1.6)", osc_nonlinear, x_ctx_non, y_ctx_non, y_true_non, y_pred_non, axes[1])
    ]
    
    # Distinct colors for each of the 10 correspondence pairs
    cmap = plt.get_cmap("tab10")
    
    for title, osc, xc, yc, y_true, y_pred, ax in systems:
        # 1. Background phase portrait orbits
        for r in np.linspace(0.4, 2.6, 7):
            init_pt = torch.tensor([[r, 0.0]], dtype=torch.float32)
            pts = [init_pt.numpy()]
            curr = init_pt
            for _ in range(120):
                curr = osc.flow_map(curr, dt=0.03, steps=4)
                pts.append(curr.numpy())
            orbit = np.concatenate(pts, axis=0)
            ax.plot(orbit[:, 0], orbit[:, 1], color="silver", lw=0.9, ls="--", alpha=0.6, zorder=1)
            
        # 2. Context observations
        xc_np = xc.squeeze(0).numpy()
        yc_np = yc.squeeze(0).numpy()
        ax.scatter(xc_np[:, 0], xc_np[:, 1], color="gray", s=18, alpha=0.35, label="Context $X_{ctx}$", zorder=2)
        ax.scatter(yc_np[:, 0], yc_np[:, 1], color="dimgray", s=18, marker="s", alpha=0.35, label="Context $Y_{ctx}$", zorder=2)
        
        # 3. 10 Correspondence pairs
        for i in range(10):
            color = cmap(i)
            xi = x_q_np[i]
            y_tr = y_true[i]
            y_pr = y_pred[i]
            
            # Initial state X_0 (Circle)
            ax.scatter(xi[0], xi[1], color=color, edgecolor="black", s=90, marker="o", lw=1.2, zorder=5)
            ax.text(xi[0] + 0.05, xi[1] + 0.05, f"$X_{{{i+1}}}$", fontsize=10, fontweight="bold", color=color, zorder=6)
            
            # True state Y_true (Filled Star)
            ax.scatter(y_tr[0], y_tr[1], color=color, edgecolor="black", s=130, marker="*", lw=1.0, zorder=5)
            
            # Model predicted state Y_pred (Hollow Square)
            ax.scatter(y_pr[0], y_pr[1], facecolor="none", edgecolor=color, s=110, marker="s", lw=2.2, zorder=5)
            
            # Flow arrow: X_0 -> Y_true (solid arrow)
            ax.annotate(
                "",
                xy=(y_tr[0], y_tr[1]),
                xytext=(xi[0], xi[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85, shrinkA=5, shrinkB=5),
                zorder=4
            )
            
            # Prediction error segment: Y_true -> Y_pred (red dotted line)
            err_dist = np.linalg.norm(y_tr - y_pr)
            ax.plot([y_tr[0], y_pr[0]], [y_tr[1], y_pr[1]], color="red", ls=":", lw=1.8, alpha=0.9, zorder=4)
            
        # Legend markers
        ax.plot([], [], "ko", ms=8, label="Query Init $X_0$")
        ax.plot([], [], "k*", ms=10, label="Ground Truth $Y_{true}$")
        ax.plot([], [], "ks", ms=8, fillstyle="none", mew=2, label="Model Predicted $\hat{Y}$")
        ax.plot([], [], "r:", lw=2, label="Error $(\hat{Y} - Y_{true})$")
        
        ax.set_title(f"{title}\n$\Phi_{{\Delta t}}: (q_0, p_0) \mapsto (q_{{\Delta t}}, p_{{\Delta t}})$", fontsize=13, fontweight="bold")
        ax.set_xlabel("Displacement $q$", fontsize=12)
        ax.set_ylabel("Momentum $p$", fontsize=12)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        
    plt.suptitle("Phase Plane 10-Point Correspondence Benchmark: Ground Truth vs. Transformer Prediction", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_path = "spring_benchmark/results/phase_correspondence_10pairs.png"
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"10-pair correspondence plot saved to {save_path}")

if __name__ == "__main__":
    plot_10_correspondences()
