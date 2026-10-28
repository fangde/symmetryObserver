import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spring_benchmark.physics.spring_system import DuffingOscillator, CoupledSpringChain8D

def generate_phase_plane_trajectories(oscillator, grid_size=9, extent=2.0, steps=250, dt=0.03):
    """
    Generates trajectories starting from concentric initial points in phase space (q, p).
    """
    radii = np.linspace(0.4, extent, grid_size)
    trajectories = []
    
    for r in radii:
        init_state = torch.tensor([[r, 0.0]], dtype=torch.float32)
        traj = [init_state.numpy()]
        curr = init_state
        for _ in range(steps):
            curr = oscillator.flow_map(curr, dt=dt, steps=5)
            traj.append(curr.numpy())
        trajectories.append(np.concatenate(traj, axis=0))
    return trajectories

def plot_linear_vs_nonlinear():
    os.makedirs("spring_benchmark/results", exist_ok=True)
    
    linear_spring = DuffingOscillator(k=1.5, beta=0.0, m=1.0)
    nonlinear_spring = DuffingOscillator(k=1.5, beta=2.0, m=1.0) # strong hardening spring
    
    traj_linear = generate_phase_plane_trajectories(linear_spring)
    traj_nonlinear = generate_phase_plane_trajectories(nonlinear_spring)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. Linear Spring Orbits
    ax = axes[0]
    for traj in traj_linear:
        ax.plot(traj[:, 0], traj[:, 1], lw=1.5)
    ax.set_title(r"Linear Spring ($\beta = 0$)" + "\nIsochronous Ellipses (Constant Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Displacement $q$", fontsize=11)
    ax.set_ylabel("Momentum $p$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    
    # 2. Nonlinear Spring Orbits
    ax = axes[1]
    for traj in traj_nonlinear:
        ax.plot(traj[:, 0], traj[:, 1], lw=1.5)
    ax.set_title(r"Nonlinear Spring ($\beta = 2.0$)" + "\nAnharmonic Orbits (Period Shrinks with Amplitude)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Displacement $q$", fontsize=11)
    ax.set_ylabel("Momentum $p$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    
    # 3. Flow Map Phase Twist: How an initial vertical line of states deforms under \Phi_{\Delta t}
    ax = axes[2]
    # Initial vertical line of test states
    q_line = np.ones(80) * 1.0
    p_line = np.linspace(-2.0, 2.0, 80)
    line_states = torch.tensor(np.stack([q_line, p_line], axis=-1), dtype=torch.float32)
    
    ax.plot(q_line, p_line, "k--", lw=2, label="Initial State Line $(t=0)$")
    
    # Evolve under linear spring
    dt_flow = 1.2
    line_linear = linear_spring.flow_map(line_states, dt=dt_flow, steps=20).numpy()
    ax.plot(line_linear[:, 0], line_linear[:, 1], color="royalblue", lw=2.5, label="Linear Flow: Pure Rigid Rotation")
    
    # Evolve under nonlinear spring
    line_nonlinear = nonlinear_spring.flow_map(line_states, dt=dt_flow, steps=20).numpy()
    ax.plot(line_nonlinear[:, 0], line_nonlinear[:, 1], color="crimson", lw=2.5, label="Nonlinear Flow: Nonlinear Shear Twist")
    
    ax.set_title(f"Phase Flow Diffeomorphism $\Phi_{{\Delta t}}$ ($t={dt_flow}s$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Displacement $q$", fontsize=11)
    ax.set_ylabel("Momentum $p$", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    save_path = "spring_benchmark/results/spring_phase_portrait_comparison.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Phase portrait plot saved to {save_path}")

if __name__ == "__main__":
    plot_linear_vs_nonlinear()
