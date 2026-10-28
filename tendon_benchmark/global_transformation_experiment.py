import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tendon_benchmark.physics.tendon_joint import TendonDrivenJoint, DirectDriveEquivalent
from diffeo_benchmark.models.transformer import DiffeoTransformer

def generate_trajectory(tendon_joint, direct_joint, dt=0.005, steps=500, freq_list=[1.2, 2.8], seed=42):
    rng = np.random.RandomState(seed)
    time_arr = np.linspace(0, steps * dt, steps)
    
    # Random multi-harmonic excitation
    u = np.zeros(steps)
    for f in freq_list:
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.4, 0.9)
        u += amp * np.sin(2 * np.pi * f * time_arr + phase)
        
    x_t = torch.zeros(1, 4) # [th_l, dth_l, th_m, dth_m]
    x_d = torch.zeros(1, 2) # [th_d, dth_d]
    
    t_hist, d_hist = [], []
    for i in range(steps):
        u_t = torch.tensor([[u[i]]], dtype=torch.float32)
        t_hist.append(x_t.squeeze(0).numpy())
        d_hist.append(x_d.squeeze(0).numpy())
        x_t = tendon_joint.step_rk4(x_t, u_t, dt)
        x_d = direct_joint.step_rk4(x_d, u_t, dt)
        
    return time_arr, np.array(t_hist), np.array(d_hist), u

def train_and_evaluate_global_transform():
    os.makedirs("tendon_benchmark/results", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Generate meta-dataset of varied tendon joints
    print("Generating diverse tendon-joint trajectories...")
    episodes = 600
    steps = 200
    dt = 0.005
    
    X_ctx, Y_ctx, X_qry, Y_qry = [], [], [], []
    
    rng = np.random.RandomState(101)
    for ep in range(episodes):
        kt = rng.uniform(40.0, 90.0)
        betat = rng.uniform(20.0, 60.0)
        mu_alpha = rng.uniform(0.15, 0.40)
        
        tj = TendonDrivenJoint(k_t=kt, beta_t=betat, mu_alpha=mu_alpha)
        dj = DirectDriveEquivalent()
        
        # Context trajectory
        _, t_ctx, d_ctx, _ = generate_trajectory(tj, dj, dt=dt, steps=64, freq_list=[1.5], seed=ep*7+1)
        # Query trajectory
        _, t_qry, d_qry, _ = generate_trajectory(tj, dj, dt=dt, steps=steps, freq_list=[1.0, 2.5], seed=ep*7+2)
        
        # We want to learn Global Transformation Phi: x_tendon (4D) -> x_direct (2D)
        # Pad direct to 4D or project to 2D
        X_ctx.append(torch.tensor(t_ctx, dtype=torch.float32)) # (64, 4)
        Y_ctx.append(torch.tensor(d_ctx, dtype=torch.float32)) # (64, 2)
        X_qry.append(torch.tensor(t_qry, dtype=torch.float32)) # (200, 4)
        Y_qry.append(torch.tensor(d_qry, dtype=torch.float32)) # (200, 2)
        
    X_ctx = torch.stack(X_ctx) # (episodes, 64, 4)
    Y_ctx = torch.stack(Y_ctx) # (episodes, 64, 2)
    X_qry = torch.stack(X_qry) # (episodes, 200, 4)
    Y_qry = torch.stack(Y_qry) # (episodes, 200, 2)
    
    # 2. Build Global Transformation Model
    # Context maps [X_tendon (4D), Y_direct (2D)] -> d_model
    # Query maps X_tendon (4D) -> Y_direct (2D)
    class GlobalDiffeoTransformer(nn.Module):
        def __init__(self, d_model=128):
            super().__init__()
            self.ctx_in = nn.Linear(4 + 2, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
            
            self.qry_in = nn.Linear(4, d_model)
            dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=3)
            
            self.out_head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Linear(128, 2) # [theta_d, dot_theta_d]
            )
            
        def forward(self, xc, yc, xq):
            # xc: (B, K, 4), yc: (B, K, 2)
            c_cat = torch.cat([xc, yc], dim=-1)
            mem = self.encoder(self.ctx_in(c_cat))
            q_emb = self.qry_in(xq)
            out = self.decoder(tgt=q_emb, memory=mem)
            pred = self.out_head(out)
            # Link angle + predicted correction
            return pred

    model = GlobalDiffeoTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print("Training Global Transformation Transformer on Tendon -> Direct-Drive...")
    epochs = 6
    batch_size = 32
    num_batches = episodes // batch_size
    
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
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {total_loss/num_batches:.5f}")
        
    # 3. Test on a completely new, unseen tendon-driven joint!
    print("\nEvaluating Global Transformation on Unseen Test Joint...")
    test_tj = TendonDrivenJoint(k_t=72.5, beta_t=45.0, mu_alpha=0.32) # novel unseen parameters
    test_dj = DirectDriveEquivalent()
    
    # Gather 64 context points
    _, test_t_ctx, test_d_ctx, _ = generate_trajectory(test_tj, test_dj, dt=0.005, steps=64, freq_list=[1.3], seed=9999)
    # Long test evaluation trajectory (600 steps = 3.0s)
    test_time, test_t_raw, test_d_real, _ = generate_trajectory(test_tj, test_dj, dt=0.005, steps=600, freq_list=[1.0, 2.7], seed=8888)
    
    model.eval()
    with torch.no_grad():
        xc = torch.tensor(test_t_ctx, dtype=torch.float32).unsqueeze(0).to(device)
        yc = torch.tensor(test_d_ctx, dtype=torch.float32).unsqueeze(0).to(device)
        xq = torch.tensor(test_t_raw, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Apply Global Transformation Phi(x_tendon)
        pred_d = model(xc, yc, xq).squeeze(0).cpu().numpy() # (600, 2)
        
    # Compute quantitative improvement
    # Raw Tendon Link vs Real Direct Drive
    raw_pos_rmse = np.sqrt(np.mean((test_t_raw[:, 0] - test_d_real[:, 0])**2))
    raw_vel_rmse = np.sqrt(np.mean((test_t_raw[:, 1] - test_d_real[:, 1])**2))
    
    # Transformed Tendon vs Real Direct Drive
    trans_pos_rmse = np.sqrt(np.mean((pred_d[:, 0] - test_d_real[:, 0])**2))
    trans_vel_rmse = np.sqrt(np.mean((pred_d[:, 1] - test_d_real[:, 1])**2))
    
    print("="*65)
    print("      GLOBAL TRANSFORMATION PERFORMANCE: TENDON -> DIRECT DRIVE   ")
    print("="*65)
    print(f"Position RMSE: Raw Tendon: {raw_pos_rmse:.4f} -> Transformed: {trans_pos_rmse:.4f} (Error reduced by {(1-trans_pos_rmse/raw_pos_rmse)*100:.1f}%)")
    print(f"Velocity RMSE: Raw Tendon: {raw_vel_rmse:.4f} -> Transformed: {trans_vel_rmse:.4f} (Error reduced by {(1-trans_vel_rmse/raw_vel_rmse)*100:.1f}%)")
    print("="*65)
    
    # 4. Generate the side-by-side comparison visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    
    # Panel 1: Position Comparison
    ax1 = axes[0]
    ax1.plot(test_time, test_d_real[:, 0], "k-", lw=2.5, label="Real Ideal Direct-Drive Target $\\theta_d(t)$ (Ground Truth)")
    ax1.plot(test_time, test_t_raw[:, 0], "r--", lw=1.6, alpha=0.8, label="Raw Tendon Joint Output $\\theta_l(t)$ (Lag & Attenuation)")
    ax1.plot(test_time, pred_d[:, 0], "b-.", lw=2.0, label="Globally Transformed $\\hat{\\theta}_d = \\Phi(x_{tendon})$ (Straightened)")
    ax1.set_ylabel("Position (rad)", fontsize=11, fontweight="bold")
    ax1.set_title("Position Alignment: Global Transformation Eliminates Compliance Lag & Hysteresis", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Panel 2: Velocity Comparison
    ax2 = axes[1]
    ax2.plot(test_time, test_d_real[:, 1], "k-", lw=2.2, label="Real Direct-Drive Velocity $\\dot{\\theta}_d(t)$")
    ax2.plot(test_time, test_t_raw[:, 1], "r--", lw=1.4, alpha=0.7, label="Raw Tendon Velocity $\\dot{\\theta}_l(t)$ (Ripple & Chattering)")
    ax2.plot(test_time, pred_d[:, 1], "b-.", lw=1.8, label="Globally Transformed Velocity $\\hat{\\dot{\\theta}}_d = \\Phi(x_{tendon})$")
    ax2.set_ylabel("Velocity (rad/s)", fontsize=11, fontweight="bold")
    ax2.set_title("Velocity Alignment: Friction Ripple & Direction-Reversal Deadband Cleared", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    # Panel 3: Instantaneous Absolute Tracking Error
    ax3 = axes[2]
    raw_err = np.abs(test_t_raw[:, 0] - test_d_real[:, 0])
    trans_err = np.abs(pred_d[:, 0] - test_d_real[:, 0])
    ax3.plot(test_time, raw_err, "r-", lw=1.5, alpha=0.7, label=f"Raw Tendon Error (Mean: {np.mean(raw_err):.4f})")
    ax3.plot(test_time, trans_err, "g-", lw=2.0, label=f"Post-Transformation Error (Mean: {np.mean(trans_err):.4f})")
    ax3.fill_between(test_time, trans_err, raw_err, color="green", alpha=0.15, label="Error Eliminated by Diffeomorphism $\\Phi$")
    ax3.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Position Abs Error (rad)", fontsize=11, fontweight="bold")
    ax3.set_title("Instantaneous Error: Transformed vs. Real", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    save_path = "tendon_benchmark/results/global_transformation_comparison.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved global transformation plot to {save_path}")

if __name__ == "__main__":
    train_and_evaluate_global_transform()
