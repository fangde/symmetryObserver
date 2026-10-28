import torch
import torch.nn as nn
import math
from typing import Optional

class DiffeoTransformer(nn.Module):
    """
    In-Context Transformer for Diffeomorphic Map Learning.
    
    Context pairs (X_i, Y_i) are encoded and processed by self-attention blocks.
    Query points X_q attend to the context via cross-attention and self-attention,
    predicting coordinates Y_q via residual displacement.
    """
    def __init__(
        self,
        dim: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        residual_output: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.residual_output = residual_output
        
        # Context embedding: maps [X, Y] (2*dim) -> d_model
        self.ctx_embed = nn.Sequential(
            nn.Linear(2 * dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Query embedding: maps X_q (dim) -> d_model
        self.qry_embed = nn.Sequential(
            nn.Linear(dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer Encoder for Context set reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Transformer Decoder for Query mapping conditioned on Context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output coordinate projection
        self.out_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim)
        )

    def forward(self, x_ctx: torch.Tensor, y_ctx: torch.Tensor, x_qry: torch.Tensor) -> torch.Tensor:
        """
        x_ctx: (B, K, D)
        y_ctx: (B, K, D)
        x_qry: (B, M, D)
        Returns:
        y_hat: (B, M, D)
        """
        # Embed context pairs
        ctx_cat = torch.cat([x_ctx, y_ctx], dim=-1) # (B, K, 2D)
        h_ctx = self.ctx_embed(ctx_cat) # (B, K, d_model)
        
        # Context relational encoding
        memory = self.encoder(h_ctx) # (B, K, d_model)
        
        # Embed query points
        h_qry = self.qry_embed(x_qry) # (B, M, d_model)
        
        # Cross-attention decoding
        out = self.decoder(tgt=h_qry, memory=memory) # (B, M, d_model)
        
        # Output coordinates
        delta = self.out_head(out) # (B, M, D)
        
        if self.residual_output:
            return x_qry + delta
        else:
            return delta
