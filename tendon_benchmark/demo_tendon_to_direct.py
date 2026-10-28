import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tendon_benchmark.physics.tendon_joint import TendonDrivenJoint, DirectDriveEquivalent

def simulate_responses():
    os.makedirs("tendon_benchmark/results", exist_ok=True)
    dt = 0.002
    T_total = 3.0
    steps = int(T_total / dt)
    time_arr = np.linspace(0, T_total, steps)
    
    # Input torque: multi-sine chirp excitation
    freq1, freq2 = 1.0, 3.5
    u_traj = 0.6 * np.sin(2 * np.pi * freq1 * time_arr) + 0.4 * np.sin(2 * np.pi * freq2 * time_arr)
    
    # Instantiate joints
    tendon_joint = TendonDrivenJoint(k_t=60.0, beta_t=40.0, mu_alpha=0.35)
    direct_joint = DirectDriveEquivalent()
    
    x_tendon = torch.zeros(1, 4)
    x_direct = torch.zeros(1, 2)
    
    tendon_history = []
    direct_history = []
    
    for t_idx in range(steps):
        u_t = torch.tensor([[u_traj[t_idx]]], dtype=torch.float32)
        tendon_history.append(x_tendon.squeeze(0).numpy())
        direct_history.append(x_direct.squeeze(0).numpy())
        
        x_tendon = tendon_joint.step_rk4(x_tendon, u_t, dt)
        x_direct = direct_joint.step_rk4(x_direct, u_t, dt)
        
    th_tendon = np.array(tendon_history) # (N, 4) -> [th_l, dth_l, th_m, dth_m]
    th_direct = np.array(direct_history) # (N, 2) -> [th_d, dth_d]
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    
    # 1. Position Trajectory: Motor vs Tendon Link vs Direct Drive
    ax = axes[0, 0]
    ax.plot(time_arr, th_direct[:, 0], "k-", lw=2.2, label=r"Direct-Drive Joint $\theta_d(t)$ (Ideal/Rigid)")
    ax.plot(time_arr, th_tendon[:, 2], "b--", lw=1.5, label=r"Tendon Motor Side $\theta_m(t)$")
    ax.plot(time_arr, th_tendon[:, 0], "r-", lw=1.8, label=r"Tendon Joint Output $\theta_l(t)$ (Lag & Attenuation)")
    ax.set_title("Position Response: Direct-Drive vs. Tendon-Driven Joint", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # 2. Velocity Trajectory: vel shows severe high-frequency oscillation & deadband!
    ax = axes[0, 1]
    ax.plot(time_arr, th_direct[:, 1], "k-", lw=2.0, label=r"Direct-Drive $\dot{\theta}_d(t)$")
    ax.plot(time_arr, th_tendon[:, 1], "r-", lw=1.5, label=r"Tendon Joint $\dot{\theta}_l(t)$")
    ax.set_title("Velocity Response: Severe Phase Shear & High-Frequency Ripple", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # 3. Phase Plane: (Angle, Velocity)
    ax = axes[1, 0]
    ax.plot(th_direct[:, 0], th_direct[:, 1], "k-", lw=2.0, label="Direct-Drive Phase Orbit")
    ax.plot(th_tendon[:, 0], th_tendon[:, 1], "r--", lw=1.2, alpha=0.85, label="Tendon-Driven Phase Orbit")
    ax.set_title(r"Phase Space $(\theta, \dot{\theta})$: Diffeomorphic Distortion", fontsize=12, fontweight="bold")
    ax.set_xlabel("Joint Angle (rad)")
    ax.set_ylabel("Joint Velocity (rad/s)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # 4. Tendon Nonlinear Deflection & Hysteresis: Torque vs Elongation
    ax = axes[1, 1]
    delta = tendon_joint.r * (th_tendon[:, 2] - th_tendon[:, 0])
    ddelta = tendon_joint.r * (th_tendon[:, 3] - th_tendon[:, 1])
    delta_t = torch.tensor(delta, dtype=torch.float32)
    ddelta_t = torch.tensor(ddelta, dtype=torch.float32)
    tension = tendon_joint.tendon_tension(delta_t, ddelta_t).numpy()
    
    ax.plot(delta, tension, "crimson", lw=1.8, label="Tendon Hysteresis Loop (Capstan + Elastic)")
    ax.plot(delta, tendon_joint.k_t * delta, "gray", ls="--", label="Linear Elastic Baseline")
    ax.set_title(r"Tendon Transmission: Elasticity & Friction Hysteresis Loop", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tendon Deflection $\Delta$ (m)")
    ax.set_ylabel("Tendon Tension (N)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    save_path = "tendon_benchmark/results/tendon_vs_direct_drive.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved tendon simulation plot to {save_path}")

if __name__ == "__main__":
    simulate_responses()
