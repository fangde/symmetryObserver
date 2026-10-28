import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from spring_benchmark.physics.spring_system import DuffingOscillator, CoupledSpringChain8D

class SpringEpisodeDataset(Dataset):
    """
    Episodes for spring dynamical systems:
    Either 1-DOF (dim=2: q, p) or 4-DOF Coupled Chain (dim=8: q1..q4, p1..p4).
    """
    def __init__(
        self,
        num_episodes: int,
        dim: int = 8,
        is_nonlinear: bool = True,
        num_context: int = 32,
        num_query: int = 128,
        dt: float = 0.5,
        seed: int = 42
    ):
        super().__init__()
        self.num_episodes = num_episodes
        self.dim = dim
        self.is_nonlinear = is_nonlinear
        self.num_context = num_context
        self.num_query = num_query
        self.dt = dt
        self.seed = seed
        
        rng = np.random.RandomState(seed)
        self.k_list = rng.uniform(0.8, 2.5, size=num_episodes)
        if is_nonlinear:
            self.beta_list = rng.uniform(0.5, 2.5, size=num_episodes)
        else:
            self.beta_list = np.zeros(num_episodes)

    def get_system(self, idx: int):
        k = float(self.k_list[idx])
        beta = float(self.beta_list[idx])
        if self.dim == 2:
            return DuffingOscillator(k=k, beta=beta, m=1.0)
        else:
            return CoupledSpringChain8D(k=k, beta=beta, m=1.0)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict:
        sys = self.get_system(idx)
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx * 31337)
        
        # Sample random initial states in phase space
        # q ~ N(0, 1), p ~ N(0, 1)
        x_ctx = torch.randn(self.num_context, self.dim, generator=rng) * 1.2
        y_ctx = sys.flow_map(x_ctx, dt=self.dt)
        
        x_qry = torch.randn(self.num_query, self.dim, generator=rng) * 1.2
        y_qry = sys.flow_map(x_qry, dt=self.dt)
        
        return {
            "x_ctx": x_ctx,
            "y_ctx": y_ctx,
            "x_qry": x_qry,
            "y_qry": y_qry,
            "k": self.k_list[idx],
            "beta": self.beta_list[idx],
            "episode_idx": idx
        }

