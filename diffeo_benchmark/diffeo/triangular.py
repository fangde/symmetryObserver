import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class TriangularDiffeoLayer:
    r"""
    Single triangular diffeomorphic layer:
    y_i = e^{s_i} x_i + \sum_{j < i} a_{ij} tanh(w_{ij} x_j + b_{ij})
    
    Guarantees:
    - det(J) = \prod_{i=1}^D e^{s_i} > 0 everywhere
    - Smooth and strictly invertible via forward substitution
    """
    def __init__(self, dim: int, s: torch.Tensor, a: torch.Tensor, w: torch.Tensor, b: torch.Tensor, perm: torch.Tensor):
        self.dim = dim
        self.s = s  # (dim,)
        self.a = a  # (dim, dim) strictly lower triangular
        self.w = w  # (dim, dim) strictly lower triangular
        self.b = b  # (dim, dim) strictly lower triangular
        self.perm = perm  # (dim,) permutation indices
        self.inv_perm = torch.argsort(perm)

    def to(self, device: torch.device):
        self.s = self.s.to(device)
        self.a = self.a.to(device)
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        self.perm = self.perm.to(device)
        self.inv_perm = self.inv_perm.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (*batch, dim)
        Returns: (*batch, dim)
        """
        # Apply permutation first
        xp = x[..., self.perm]
        
        # We can compute y iteratively or triangularly
        # y_i = exp(s_i)*xp_i + sum_{j<i} a_{ij} * tanh(w_{ij}*xp_j + b_{ij})
        # Efficient vectorized computation across batch
        orig_shape = xp.shape
        flat_x = xp.reshape(-1, self.dim) # (B, D)
        B, D = flat_x.shape
        
        # Precompute coupling contributions: for each i, j < i
        # flat_x: (B, 1, D)
        x_expanded = flat_x.unsqueeze(1) # (B, 1, D)
        # w: (1, D, D), b: (1, D, D)
        arg = self.w.unsqueeze(0) * x_expanded + self.b.unsqueeze(0) # (B, D, D)
        tanh_term = self.a.unsqueeze(0) * torch.tanh(arg) # (B, D, D)
        coupling = torch.sum(tanh_term, dim=-1) # (B, D), since a is strictly lower triangular, sum is over j < i
        
        y = flat_x * torch.exp(self.s).unsqueeze(0) + coupling
        return y.reshape(orig_shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (*batch, dim)
        Returns: (*batch, dim)
        """
        orig_shape = y.shape
        flat_y = y.reshape(-1, self.dim) # (B, D)
        B, D = flat_y.shape
        
        flat_x = torch.zeros_like(flat_y)
        exp_neg_s = torch.exp(-self.s)
        
        for i in range(D):
            if i == 0:
                coupling_i = 0.0
            else:
                # x up to i-1 are already solved: (B, i)
                prev_x = flat_x[:, :i] # (B, i)
                w_row = self.w[i, :i].unsqueeze(0) # (1, i)
                b_row = self.b[i, :i].unsqueeze(0) # (1, i)
                a_row = self.a[i, :i].unsqueeze(0) # (1, i)
                coupling_i = torch.sum(a_row * torch.tanh(w_row * prev_x + b_row), dim=-1) # (B,)
                
            flat_x[:, i] = (flat_y[:, i] - coupling_i) * exp_neg_s[i]
            
        # Invert permutation
        x = flat_x[:, self.inv_perm]
        return x.reshape(orig_shape)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes exact analytic Jacobian J = dy / dx at point x.
        x: (*batch, dim)
        Returns: (*batch, dim, dim) where J[..., i, j] = dy_i / dx_j
        """
        orig_shape = x.shape[:-1]
        xp = x[..., self.perm]
        flat_x = xp.reshape(-1, self.dim)
        B, D = flat_x.shape
        
        # d y_flat / d xp
        J_tri = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        
        # Diagonal: exp(s_i)
        for i in range(D):
            J_tri[:, i, i] = torch.exp(self.s[i])
            
        # Lower triangular: d/dxp_j [ a_ij * tanh(w_ij * xp_j + b_ij) ]
        # = a_ij * w_ij * (1 - tanh^2(w_ij * xp_j + b_ij))
        x_expanded = flat_x.unsqueeze(1) # (B, 1, D)
        arg = self.w.unsqueeze(0) * x_expanded + self.b.unsqueeze(0)
        sech2 = 1.0 - torch.tanh(arg) ** 2
        d_coupling = self.a.unsqueeze(0) * self.w.unsqueeze(0) * sech2 # (B, D, D)
        
        # Mask strictly lower triangular
        tril_mask = torch.tril(torch.ones(D, D, device=x.device), diagonal=-1).bool()
        J_tri[:, tril_mask] = d_coupling[:, tril_mask]
        
        # Incorporate permutation: y = f(P x) => dy/dx = (dy/dxp) * P
        # Permutation matrix P: P[k, l] = 1 if perm[k] == l
        P = torch.zeros(D, D, device=x.device, dtype=x.dtype)
        P[torch.arange(D, device=x.device), self.perm] = 1.0
        
        J = torch.matmul(J_tri, P.unsqueeze(0)) # (B, D, D)
        return J.reshape(*orig_shape, D, D)


class CompositeDiffeomorphism:
    r"""
    \Phi_z = \phi_L \circ \phi_{L-1} \circ ... \circ \phi_1
    """
    def __init__(self, dim: int, layers: List[TriangularDiffeoLayer]):
        self.dim = dim
        self.layers = layers

    def to(self, device: torch.device):
        for layer in self.layers:
            layer.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        out = y
        for layer in reversed(self.layers):
            out = layer.inverse(out)
        return out

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Chain rule: J_{\phi_L \circ ... \circ \phi_1}(x) = J_{\phi_L}(x_{L-1}) * ... * J_{\phi_1}(x)
        """
        curr = x
        Js = []
        for layer in self.layers:
            J_layer = layer.jacobian(curr)
            Js.append(J_layer)
            curr = layer.forward(curr)
            
        # Product of Jacobians: J_L * J_{L-1} * ... * J_1
        total_J = Js[0]
        for J_next in Js[1:]:
            total_J = torch.matmul(J_next, total_J)
        return total_J
