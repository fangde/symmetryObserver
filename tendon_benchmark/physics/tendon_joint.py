import torch
import numpy as np

class TendonDrivenJoint:
    """
    Tendon-driven joint model:
      Motor inertia: J_m, damping: b_m
      Link inertia: J_l, damping: b_l, gravity: m*g*l
      Tendon pulley radius: r
      Tendon stiffness: k_t (linear) + beta_t (nonlinear hardening)
      Capstan sheath friction: mu * alpha (curvature angle alpha)
    
    State: x = [theta_l, dot_theta_l, theta_m, dot_theta_m] (4D)
    """
    def __init__(
        self,
        J_l: float = 0.05,
        b_l: float = 0.02,
        mgl: float = 0.8,
        J_m: float = 0.01,
        b_m: float = 0.01,
        r: float = 0.02,
        k_t: float = 80.0,
        beta_t: float = 50.0,
        mu_alpha: float = 0.25 # Capstan friction coefficient * wrap angle
    ):
        self.J_l = J_l
        self.b_l = b_l
        self.mgl = mgl
        self.J_m = J_m
        self.b_m = b_m
        self.r = r
        self.k_t = k_t
        self.beta_t = beta_t
        self.mu_alpha = mu_alpha

    def tendon_tension(self, delta: torch.Tensor, ddelta: torch.Tensor) -> torch.Tensor:
        """
        delta = r * (theta_m - theta_l) is tendon elongation.
        Tendon can pull in both directions in an antagonist / pre-tensioned configuration,
        or unidirectional with pretension T_0.
        Here we model pre-tensioned antagonistic pair net torque:
          tau_t = k_t * delta + beta_t * (delta^3)
        With Capstan routing friction:
          T_trans = tau_t * exp(-mu_alpha * tanh(10 * ddelta))
        """
        raw_tau = self.k_t * delta + self.beta_t * (delta ** 3)
        # Smooth sign via tanh
        friction_factor = torch.exp(-self.mu_alpha * torch.tanh(15.0 * ddelta))
        return raw_tau * friction_factor

    def derivatives(self, state: torch.Tensor, u_motor: torch.Tensor) -> torch.Tensor:
        """
        state: (*, 4) -> [th_l, dth_l, th_m, dth_m]
        u_motor: (*, 1) or scalar
        """
        th_l = state[..., 0:1]
        dth_l = state[..., 1:2]
        th_m = state[..., 2:3]
        dth_m = state[..., 3:4]
        
        delta = self.r * (th_m - th_l)
        ddelta = self.r * (dth_m - dth_l)
        
        tau_tendon = self.tendon_tension(delta, ddelta)
        
        # Link dynamics: J_l * d2th_l + b_l * dth_l + mgl * sin(th_l) = tau_tendon
        d2th_l = (tau_tendon - self.b_l * dth_l - self.mgl * torch.sin(th_l)) / self.J_l
        
        # Motor dynamics: J_m * d2th_m + b_m * dth_m = u_motor - tau_tendon
        d2th_m = (u_motor - tau_tendon - self.b_m * dth_m) / self.J_m
        
        return torch.cat([dth_l, d2th_l, dth_m, d2th_m], dim=-1)

    def step_rk4(self, state: torch.Tensor, u_motor: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.derivatives(state, u_motor)
        k2 = self.derivatives(state + 0.5 * dt * k1, u_motor)
        k3 = self.derivatives(state + 0.5 * dt * k2, u_motor)
        k4 = self.derivatives(state + dt * k3, u_motor)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class DirectDriveEquivalent:
    """
    Ideal Direct-Drive Joint:
      J_eq * d2th + b_eq * dth + mgl * sin(th) = u_motor
    where J_eq = J_l + J_m, b_eq = b_l + b_m.
    Completely rigid, no compliance delay, no cable sheath friction!
    """
    def __init__(self, J_eq: float = 0.06, b_eq: float = 0.03, mgl: float = 0.8):
        self.J_eq = J_eq
        self.b_eq = b_eq
        self.mgl = mgl

    def derivatives(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        th = state[..., 0:1]
        dth = state[..., 1:2]
        d2th = (u - self.b_eq * dth - self.mgl * torch.sin(th)) / self.J_eq
        return torch.cat([dth, d2th], dim=-1)

    def step_rk4(self, state: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.derivatives(state, u)
        k2 = self.derivatives(state + 0.5 * dt * k1, u)
        k3 = self.derivatives(state + 0.5 * dt * k2, u)
        k4 = self.derivatives(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
