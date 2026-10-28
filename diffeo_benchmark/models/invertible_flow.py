import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class ConditionalAffineCouplingLayer(nn.Module):
    """
    Conditional Affine Coupling Layer (RealNVP style) conditioned on transformation code z.
    Input x is split into x1 and x2:
      y1 = x1
      y2 = x2 * exp(s(x1, z)) + t(x1, z)
    """
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 128, mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        
        # If mask is None, standard split first half / second half
        if mask is None:
            mask = torch.cat([torch.ones(self.split_dim), torch.zeros(dim - self.split_dim)])
        self.register_buffer("mask", mask)
        
        # Network to predict scale and translation from x1 and cond_dim
        # x_masked has size dim, cond has size cond_dim
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * 2)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, D)
        z: (B, cond_dim) or (B, M, cond_dim)
        """
        B, M, D = x.shape
        if z.dim() == 2:
            z_exp = z.unsqueeze(1).expand(B, M, -1)
        else:
            z_exp = z
            
        x_masked = x * self.mask # keep active coordinates
        net_in = torch.cat([x_masked, z_exp], dim=-1)
        st = self.net(net_in) # (B, M, 2 * (D - split_dim))
        
        s, t = st.chunk(2, dim=-1)
        # Use tanh on scale to maintain stability
        s = torch.tanh(s) * 1.5
        
        # Apply transformation only on inverted mask coordinates
        inv_mask_idx = (self.mask == 0).nonzero(as_tuple=True)[0]
        y = x.clone()
        y[..., inv_mask_idx] = x[..., inv_mask_idx] * torch.exp(s) + t
        return y

    def inverse(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Closed-form inverse:
          x1 = y1
          x2 = (y2 - t(y1, z)) * exp(-s(y1, z))
        """
        B, M, D = y.shape
        if z.dim() == 2:
            z_exp = z.unsqueeze(1).expand(B, M, -1)
        else:
            z_exp = z
            
        y_masked = y * self.mask
        net_in = torch.cat([y_masked, z_exp], dim=-1)
        st = self.net(net_in)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 1.5
        
        inv_mask_idx = (self.mask == 0).nonzero(as_tuple=True)[0]
        x = y.clone()
        x[..., inv_mask_idx] = (y[..., inv_mask_idx] - t) * torch.exp(-s)
        return x


class InvertibleFlowTransformer(nn.Module):
    """
    Model C: Transformer Context Inference + Explicit Invertible Normalizing Flow Head.
    
    Transformer encodes Context (X_i, Y_i) into a global conditioning code z.
    Query points are transformed by Conditional Affine Coupling Layers.
    Guaranteed: Strictly invertible diffeomorphism everywhere!
    """
    def __init__(
        self,
        dim: int = 8,
        cond_dim: int = 128,
        num_enc_layers: int = 3,
        num_flow_layers: int = 4,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        
        # Context Encoder (Transformer)
        self.ctx_embed = nn.Sequential(
            nn.Linear(2 * dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cond_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        
        # Summary pooling projection
        self.summary_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # Conditional Invertible Coupling layers
        self.flow_layers = nn.ModuleList()
        for l in range(num_flow_layers):
            # Alternating masks
            mask = torch.zeros(dim)
            if l % 2 == 0:
                mask[: dim // 2] = 1.0
            else:
                mask[dim // 2 :] = 1.0
            self.flow_layers.append(ConditionalAffineCouplingLayer(dim, cond_dim, hidden_dim, mask))

    def encode_context(self, x_ctx: torch.Tensor, y_ctx: torch.Tensor) -> torch.Tensor:
        ctx_cat = torch.cat([x_ctx, y_ctx], dim=-1) # (B, K, 2D)
        h = self.ctx_embed(ctx_cat)
        mem = self.context_encoder(h)
        # Mean pool to single latent transformation descriptor z
        z = self.summary_proj(torch.mean(mem, dim=1)) # (B, cond_dim)
        return z

    def forward(self, x_ctx: torch.Tensor, y_ctx: torch.Tensor, x_qry: torch.Tensor) -> torch.Tensor:
        z = self.encode_context(x_ctx, y_ctx)
        curr = x_qry
        for layer in self.flow_layers:
            curr = layer.forward(curr, z)
        return curr

    def inverse(self, x_ctx: torch.Tensor, y_ctx: torch.Tensor, y_qry: torch.Tensor) -> torch.Tensor:
        z = self.encode_context(x_ctx, y_ctx)
        curr = y_qry
        for layer in reversed(self.flow_layers):
            curr = layer.inverse(curr, z)
        return curr
