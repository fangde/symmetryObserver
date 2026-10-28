import torch
import numpy as np
from typing import Tuple, Optional

class DuffingOscillator:
    """
    1-DOF Spring System:
    Linear spring: \beta = 0
    Nonlinear spring: \beta != 0 (hardening if \beta > 0, softening if \beta < 0)
    
    Hamiltonian:
      H(q, p) = 0.5 * p^2 / m + 0.5 * k * q^2 + 0.25 * \beta * q^4
    Equations of motion:
      dq/dt = p / m
      dp/dt = - k * q - \beta * q^3
    """
    def __init__(self, k: float = 1.0, beta: float = 0.0, m: float = 1.0):
        self.k = k
        self.beta = beta
        self.m = m

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (*, 2), where state[..., 0] = q, state[..., 1] = p
        """
        q = state[..., 0]
        p = state[..., 1]
        kin = 0.5 * (p ** 2) / self.m
        pot = 0.5 * self.k * (q ** 2) + 0.25 * self.beta * (q ** 4)
        return kin + pot

    def derivatives(self, state: torch.Tensor) -> torch.Tensor:
        q = state[..., 0]
        p = state[..., 1]
        dq = p / self.m
        dp = - self.k * q - self.beta * (q ** 3)
        return torch.stack([dq, dp], dim=-1)

    def step_rk4(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Classic RK4 integration step (or symplectic step)
        """
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def flow_map(self, state: torch.Tensor, dt: float, steps: int = 10) -> torch.Tensor:
        """
        Computes flow map \Phi_{\Delta t}(x_0) by integrating 'steps' sub-steps of size dt/steps.
        This represents the exact area-preserving diffeomorphism of the phase space flow.
        """
        curr = state.clone()
        sub_dt = dt / steps
        for _ in range(steps):
            curr = self.step_rk4(curr, sub_dt)
        return curr


class CoupledSpringChain8D:
    """
    4-Mass Coupled Spring Chain (8D Phase Space: q_1..q_4, p_1..p_4)
    Fixed-boundary or periodic chain of coupled springs.
    
    Potential energy:
      V(q) = \sum_{i=0}^4 [ 0.5 * k * (q_{i+1} - q_i)^2 + 0.25 * \beta * (q_{i+1} - q_i)^4 ]
      where q_0 = q_5 = 0 (fixed boundaries)
    
    When \beta = 0: Completely linear coupled harmonic oscillators.
    When \beta > 0: FPUT nonlinear lattice with energy exchange among normal modes.
    """
    def __init__(self, k: float = 1.0, beta: float = 0.5, m: float = 1.0):
        self.dim = 8
        self.num_masses = 4
        self.k = k
        self.beta = beta
        self.m = m

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (*, 8) -> (q_1..q_4, p_1..p_4)
        """
        q = state[..., :4]
        p = state[..., 4:]
        
        # Kinetic energy
        kin = 0.5 * torch.sum(p ** 2, dim=-1) / self.m
        
        # Fixed boundaries: q_0 = 0, q_5 = 0
        pad_shape = list(q.shape)
        pad_shape[-1] = 1
        zero_pad = torch.zeros(pad_shape, device=state.device, dtype=state.dtype)
        q_padded = torch.cat([zero_pad, q, zero_pad], dim=-1) # (B, 6)
        
        diff = q_padded[..., 1:] - q_padded[..., :-1] # (B, 5)
        pot = torch.sum(0.5 * self.k * (diff ** 2) + 0.25 * self.beta * (diff ** 4), dim=-1)
        return kin + pot

    def derivatives(self, state: torch.Tensor) -> torch.Tensor:
        q = state[..., :4]
        p = state[..., 4:]
        
        dq = p / self.m # (B, 4)
        
        # Pad q with boundaries
        pad_shape = list(q.shape)
        pad_shape[-1] = 1
        zero_pad = torch.zeros(pad_shape, device=state.device, dtype=state.dtype)
        q_padded = torch.cat([zero_pad, q, zero_pad], dim=-1) # (B, 6)
        
        # Force on mass i from left and right spring
        # F_i = -dV/dq_i = - [ -k(q_i - q_{i-1}) - \beta(q_i - q_{i-1})^3 + k(q_{i+1} - q_i) + \beta(q_{i+1} - q_i)^3 ]
        diff_left = q_padded[..., 1:5] - q_padded[..., 0:4]
        diff_right = q_padded[..., 2:6] - q_padded[..., 1:5]
        
        force_left = - (self.k * diff_left + self.beta * (diff_left ** 3))
        force_right = self.k * diff_right + self.beta * (diff_right ** 3)
        dp = force_left + force_right # (B, 4)
        
        return torch.cat([dq, dp], dim=-1)

    def flow_map(self, state: torch.Tensor, dt: float = 0.5, steps: int = 20) -> torch.Tensor:
        """
        Symplectic / RK4 high-accuracy flow map \Phi_{\Delta t}(x)
        """
        curr = state.clone()
        sub_dt = dt / steps
        for _ in range(steps):
            k1 = self.derivatives(curr)
            k2 = self.derivatives(curr + 0.5 * sub_dt * k1)
            k3 = self.derivatives(curr + 0.5 * sub_dt * k2)
            k4 = self.derivatives(curr + sub_dt * k3)
            curr = curr + (sub_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return curr
