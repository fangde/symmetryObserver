import torch
import torch.nn as nn
from typing import Tuple, Dict

class AffineOrthogonalProjector(nn.Module):
    """
    Orthogonal Projection Operator.
    Decomposes any displacement field v(X) into:
      v(X) = v_affine(X) + v_lddmm_perp(X)
    such that v_lddmm_perp is strictly orthogonal to the affine subspace:
      < v_affine, v_lddmm_perp > = 0
      < 1, v_lddmm_perp > = 0  (zero translation)
      < X_j, v_lddmm_perp > = 0 (zero linear strain / rotation)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: (B, N, D) query positions
        V: (B, N, D) predicted displacements from LDDMM head
        
        Returns:
          V_affine_leak: (B, N, D) affine component leaked into V
          V_lddmm_perp:  (B, N, D) strictly orthogonal non-rigid deformation
        """
        B, N, D = X.shape
        device = X.device
        dtype = X.dtype
        
        # Design matrix Phi = [X, 1] of size (B, N, D + 1)
        ones = torch.ones(B, N, 1, device=device, dtype=dtype)
        Phi = torch.cat([X, ones], dim=-1) # (B, N, D + 1)
        
        # Least squares projection of V onto Phi:
        # W = (Phi^T Phi + eps I)^-1 Phi^T V -> size (B, D + 1, D)
        Phi_T = Phi.transpose(1, 2) # (B, D + 1, N)
        Gram = torch.bmm(Phi_T, Phi) # (B, D + 1, D + 1)
        
        # Regularization for numerical stability
        reg = torch.eye(D + 1, device=device, dtype=dtype).unsqueeze(0) * self.eps
        Gram_inv = torch.linalg.inv(Gram + reg) # (B, D + 1, D + 1)
        
        # Optimal affine parameters contained in V
        W_affine = torch.bmm(Gram_inv, torch.bmm(Phi_T, V)) # (B, D + 1, D)
        
        # Affine component inside V
        V_affine_leak = torch.bmm(Phi, W_affine) # (B, N, D)
        
        # Orthogonal non-rigid component: V_lddmm_perp = V - V_affine_leak
        V_lddmm_perp = V - V_affine_leak # (B, N, D)
        
        return V_affine_leak, V_lddmm_perp


class CascadeDiffeoTransformer(nn.Module):
    r"""
    Cascade Transformer with Dual Heads:
      1. Affine Head: \Phi_{affine}(x) = x A^T + b
      2. LDDMM Head: \Phi_{lddmm}(x) = v_{lddmm}^\perp(x)
      
    Total transformation:
      y = x A^T + b + v_{lddmm}^\perp(x)
      
    Guaranteed:
      The LDDMM field is strictly orthogonal to the affine space!
      No affine scaling, shear, or rotation leaks into the non-rigid head.
    """
    def __init__(
        self,
        in_dim: int = 4,   # e.g., Tendon state: [th_l, dth_l, th_m, dth_m]
        out_dim: int = 2,  # e.g., Direct-Drive state: [th_d, dth_d]
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_model = d_model
        
        # Context encoder: [X_in, Y_out] -> d_model
        self.ctx_proj = nn.Sequential(
            nn.Linear(in_dim + out_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        
        # Context summary pooling for global Affine parameters
        self.affine_summary = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, out_dim * in_dim + out_dim) # A: (out_dim, in_dim), b: (out_dim)
        )
        
        # Query encoder for non-rigid LDDMM field
        self.qry_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        
        # LDDMM displacement head
        self.lddmm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, out_dim)
        )
        
        # Orthogonal projector ensuring <Affine, LDDMM> = 0
        self.projector = AffineOrthogonalProjector()

    def forward(self, x_ctx: torch.Tensor, y_ctx: torch.Tensor, x_qry: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_ctx: (B, K, in_dim)
        y_ctx: (B, K, out_dim)
        x_qry: (B, N, in_dim)
        
        Returns dict containing:
          - 'y_total': final transformed coordinates
          - 'y_affine': affine head output
          - 'v_lddmm': strictly orthogonal LDDMM deformation field
          - 'orthogonality_inner_prod': empirical inner product <v_affine, v_lddmm>
        """
        B, N, Din = x_qry.shape
        Dout = self.out_dim
        
        # 1. Encode Context
        ctx_cat = torch.cat([x_ctx, y_ctx], dim=-1)
        mem = self.encoder(self.ctx_proj(ctx_cat)) # (B, K, d_model)
        
        # 2. Head 1: Affine Transformation Parameters (A, b)
        mem_pooled = torch.mean(mem, dim=1) # (B, d_model)
        affine_params = self.affine_summary(mem_pooled) # (B, Dout * Din + Dout)
        
        A_flat = affine_params[:, :Dout * Din]
        A = A_flat.view(B, Dout, Din) # (B, Dout, Din)
        b = affine_params[:, Dout * Din:].unsqueeze(1) # (B, 1, Dout)
        
        # Compute Affine Mapping: y_affine = x_qry A^T + b
        # x_qry: (B, N, Din), A.transpose(1, 2): (B, Din, Dout)
        y_affine = torch.bmm(x_qry, A.transpose(1, 2)) + b # (B, N, Dout)
        
        # 3. Head 2: LDDMM Non-rigid Field
        q_emb = self.qry_proj(x_qry) # (B, N, d_model)
        out_dec = self.decoder(tgt=q_emb, memory=mem) # (B, N, d_model)
        v_raw = self.lddmm_head(out_dec) # (B, N, Dout)
        
        # 4. Enforce STRICT ORTHOGONALITY: Project v_raw onto orthogonal complement of Affine space of x_qry (all Din dimensions)
        v_leak, v_lddmm_perp = self.projector(x_qry, v_raw)
        
        # 5. Cascade Combination
        # Any affine leak from LDDMM head is merged into the affine output, so total is exact!
        y_affine_effective = y_affine + v_leak
        y_total = y_affine_effective + v_lddmm_perp
        
        # Compute empirical orthogonality inner product <y_affine - x_ref, v_lddmm_perp>
        inner_prod = torch.sum(y_affine_effective * v_lddmm_perp, dim=[1, 2]) / N
        
        return {
            "y_total": y_total,
            "y_affine": y_affine_effective,
            "v_lddmm": v_lddmm_perp,
            "A": A,
            "b": b.squeeze(1),
            "orthogonality_inner_prod": inner_prod
        }
