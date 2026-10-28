import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict

from tendon_benchmark.physics.tendon_joint import TendonDrivenJoint, DirectDriveEquivalent

def generate_excitation_signal(T_total: float, dt: float, signal_type: str = "chirp", rng: np.random.RandomState = None):
    steps = int(T_total / dt)
    time_arr = np.linspace(0, T_total, steps)
    if rng is None:
        rng = np.random.RandomState(42)
        
    if signal_type == "chirp":
        # Frequency sweep from f0 to f1
        f0, f1 = rng.uniform(0.5, 1.0), rng.uniform(2.5, 4.5)
        # Chirp formula: sin(2*pi * (f0 + (f1-f0)/(2*T) * t) * t)
        phase = 2 * np.pi * (f0 + (f1 - f0) / (2 * T_total) * time_arr) * time_arr
        amp = rng.uniform(0.5, 0.9)
        u = amp * np.sin(phase)
    elif signal_type == "multisine":
        u = np.zeros(steps)
        for _ in range(4):
            freq = rng.uniform(0.5, 4.0)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.2, 0.4)
            u += amp * np.sin(2 * np.pi * freq * time_arr + phase)
    elif signal_type == "prbs":
        # Pseudo-Random Binary / Step Sequence (filtered for continuity)
        num_switches = int(T_total * 4) # switches every 0.25s
        switch_times = np.sort(rng.choice(steps, size=num_switches, replace=False))
        u_raw = np.zeros(steps)
        curr_val = rng.uniform(-0.8, 0.8)
        last_t = 0
        for sw in switch_times:
            u_raw[last_t:sw] = curr_val
            curr_val = rng.uniform(-0.8, 0.8)
            last_t = sw
        u_raw[last_t:] = curr_val
        # Low-pass filter to smooth torque
        window = int(0.04 / dt)
        u = np.convolve(u_raw, np.ones(window)/window, mode="same")
    else:
        u = 0.6 * np.sin(2 * np.pi * 1.5 * time_arr)
        
    return time_arr, u

class TendonTrajectoryDataset(Dataset):
    """
    Trajectory-based Dataset for Tendon-Driven Joint System Identification.
    Each item represents one physical joint evaluated on CONTINUOUS trajectories:
      - ctx_traj: (T_ctx, 5) -> [u, th_l, dth_l, th_m, dth_m]
      - qry_u: (T_qry, 1) -> input torque sequence for query
      - qry_tendon: (T_qry, 4) -> ground truth tendon joint states [th_l, dth_l, th_m, dth_m]
      - qry_direct: (T_qry, 2) -> ground truth direct drive states [th_d, dth_d]
    """
    def __init__(
        self,
        num_episodes: int,
        T_ctx: float = 1.0,
        T_qry: float = 2.0,
        dt: float = 0.01,
        seed: int = 42
    ):
        super().__init__()
        self.num_episodes = num_episodes
        self.T_ctx = T_ctx
        self.T_qry = T_qry
        self.dt = dt
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Sample joint parameter distributions
        self.kt_list = self.rng.uniform(40.0, 95.0, size=num_episodes)
        self.betat_list = self.rng.uniform(20.0, 70.0, size=num_episodes)
        self.mu_list = self.rng.uniform(0.15, 0.45, size=num_episodes)

    def __len__(self) -> int:
        return self.num_episodes

    def simulate_rollout(self, tj: TendonDrivenJoint, dj: DirectDriveEquivalent, u_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        steps = len(u_seq)
        xt = torch.zeros(1, 4)
        xd = torch.zeros(1, 2)
        
        t_hist, d_hist = [], []
        for i in range(steps):
            t_hist.append(xt.squeeze(0).numpy())
            d_hist.append(xd.squeeze(0).numpy())
            u_t = torch.tensor([[u_seq[i]]], dtype=torch.float32)
            xt = tj.step_rk4(xt, u_t, self.dt)
            xd = dj.step_rk4(xd, u_t, self.dt)
            
        return np.array(t_hist), np.array(d_hist)

    def __getitem__(self, idx: int) -> dict:
        ep_rng = np.random.RandomState(self.seed + idx * 7919)
        tj = TendonDrivenJoint(k_t=float(self.kt_list[idx]), beta_t=float(self.betat_list[idx]), mu_alpha=float(self.mu_list[idx]))
        dj = DirectDriveEquivalent()
        
        # 1. Context: continuous chirp excitation trajectory
        _, u_ctx = generate_excitation_signal(self.T_ctx, self.dt, signal_type="chirp", rng=ep_rng)
        t_ctx, d_ctx = self.simulate_rollout(tj, dj, u_ctx)
        # ctx_features: [u(t), th_l(t), dth_l(t), th_m(t), dth_m(t)]
        ctx_traj = np.concatenate([u_ctx[:, None], t_ctx], axis=-1)
        
        # 2. Query: continuous multi-sine excitation trajectory
        _, u_qry = generate_excitation_signal(self.T_qry, self.dt, signal_type="multisine", rng=ep_rng)
        t_qry, d_qry = self.simulate_rollout(tj, dj, u_qry)
        
        return {
            "ctx_traj": torch.tensor(ctx_traj, dtype=torch.float32),
            "ctx_direct": torch.tensor(d_ctx, dtype=torch.float32),
            "qry_u": torch.tensor(u_qry[:, None], dtype=torch.float32),
            "qry_tendon": torch.tensor(t_qry, dtype=torch.float32),
            "qry_direct": torch.tensor(d_qry, dtype=torch.float32),
            "kt": self.kt_list[idx],
            "mu": self.mu_list[idx]
        }
