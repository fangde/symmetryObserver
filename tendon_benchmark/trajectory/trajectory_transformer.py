import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TrajectoryDiffeoTransformer(nn.Module):
    """
    Trajectory-based In-Context Transformer for Tendon -> Direct-Drive Identification.
    
    Inputs:
      - ctx_traj: (B, T_ctx, 5) -> [u, th_l, dth_l, th_m, dth_m]
      - qry_tendon: (B, T_qry, 4) -> [th_l, dth_l, th_m, dth_m]
    Outputs:
      - pred_direct: (B, T_qry, 2) -> [th_d, dth_d]
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        
        # Temporal encoders with 1D convolution for smooth gradient capture
        self.ctx_conv = nn.Sequential(
            nn.Conv1d(5, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Query encoder
        self.qry_conv = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True
        )
        self.trajectory_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output head
        self.out_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 2)
        )

    def forward(self, ctx_traj: torch.Tensor, qry_tendon: torch.Tensor) -> torch.Tensor:
        # ctx_traj: (B, T_ctx, 5) -> transpose for conv1d: (B, 5, T_ctx)
        c_conv = self.ctx_conv(ctx_traj.transpose(1, 2)).transpose(1, 2) # (B, T_ctx, d_model)
        c_pe = self.pos_enc(c_conv)
        memory = self.context_encoder(c_pe) # (B, T_ctx, d_model)
        
        # qry_tendon: (B, T_qry, 4) -> (B, 4, T_qry)
        q_conv = self.qry_conv(qry_tendon.transpose(1, 2)).transpose(1, 2) # (B, T_qry, d_model)
        q_pe = self.pos_enc(q_conv)
        
        # Cross-attention decoding across trajectory
        out = self.trajectory_decoder(tgt=q_pe, memory=memory) # (B, T_qry, d_model)
        pred_direct = self.out_head(out) # (B, T_qry, 2)
        return pred_direct
