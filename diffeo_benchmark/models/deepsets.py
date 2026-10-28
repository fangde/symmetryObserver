import torch
import torch.nn as nn

class DeepSetsMLP(nn.Module):
    r"""
    DeepSets + MLP Baseline for In-Context Operator Learning.
    
    1. Each context pair (X_i, Y_i) is encoded into a latent vector via an encoder MLP.
    2. Context vectors are aggregated by permutation-invariant mean pooling: z = 1/K \sum \psi([X_i, Y_i]).
    3. Query coordinates X_q are combined with z through a decoder MLP to predict Y_q.
    """
    def __init__(
        self,
        dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_enc_layers: int = 3,
        num_dec_layers: int = 4,
        residual_output: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.residual_output = residual_output
        
        # Context pair encoder
        enc_layers = [nn.Linear(2 * dim, hidden_dim), nn.GELU()]
        for _ in range(num_enc_layers - 2):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        enc_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder MLP: [X_q, z] -> Y_q
        dec_layers = [nn.Linear(dim + latent_dim, hidden_dim), nn.GELU()]
        for _ in range(num_dec_layers - 2):
            dec_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        dec_layers.append(nn.Linear(hidden_dim, dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x_ctx: torch.Tensor, y_ctx: torch.Tensor, x_qry: torch.Tensor) -> torch.Tensor:
        """
        x_ctx: (B, K, D)
        y_ctx: (B, K, D)
        x_qry: (B, M, D)
        Returns:
        y_hat: (B, M, D)
        """
        ctx_cat = torch.cat([x_ctx, y_ctx], dim=-1) # (B, K, 2D)
        z_tokens = self.encoder(ctx_cat) # (B, K, latent_dim)
        z = torch.mean(z_tokens, dim=1) # (B, latent_dim)
        
        # Expand z to match query batch
        B, M, D = x_qry.shape
        z_expanded = z.unsqueeze(1).expand(B, M, -1) # (B, M, latent_dim)
        
        dec_in = torch.cat([x_qry, z_expanded], dim=-1) # (B, M, D + latent_dim)
        delta = self.decoder(dec_in) # (B, M, D)
        
        if self.residual_output:
            return x_qry + delta
        else:
            return delta
