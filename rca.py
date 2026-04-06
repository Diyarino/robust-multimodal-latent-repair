"""
Implementation of the Residual-Contrastive Attention (RCA) module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualContrastiveAttention(nn.Module):
    """
    RCA Module: Detects anomalies using the Local-Global Ratio (LGR) and 
    corrects them via an Inverse-Residual Gate.
    """
    def __init__(self, embed_dim: int = 288, num_heads: int = 8, dropout: float = 0.1):
        super(ResidualContrastiveAttention, self).__init__()
        
        # Global Mixing via Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                         dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Learnable sensitivity parameter (\beta) for anomaly scoring
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Moving average for the LGR expectation (\mu_\lambda)
        self.register_buffer('mu_lambda', torch.tensor(1.0))

    def forward(self, z: torch.Tensor):
        """
        Args:
            z (torch.Tensor): Unified latent sequence of shape (Batch, Tokens, D)
        Returns:
            Y (torch.Tensor): Repaired latent sequence.
            alpha (torch.Tensor): Estimated anomaly probability scores.
        """
        # 1. Attention Path (Global Context)
        z_normed = self.layer_norm(z)
        mha_out, _ = self.mha(z_normed, z_normed, z_normed)
        
        # 2. Local-Global Ratio (LGR) Calculation
        # L2 norm along the feature dimension (D)
        norm_z = torch.norm(z, p=2, dim=-1)
        norm_mha = torch.norm(mha_out, p=2, dim=-1)
        
        # \lambda_t
        lgr = norm_z / (norm_mha + 1e-6)
        
        # Update moving average \mu_\lambda during training
        if self.training:
            with torch.no_grad():
                self.mu_lambda.copy_(0.9 * self.mu_lambda + 0.1 * lgr.mean())
                
        # 3. Anomaly Score Calculation (\alpha_t)
        alpha = torch.sigmoid(self.beta * (lgr - self.mu_lambda))
        
        # Ensure alpha broadcasts correctly over feature dimension (B, T, 1)
        alpha_expanded = alpha.unsqueeze(-1)
        
        # 4. Inverse-Residual Gate
        # Suppress local residual and amplify global context proportional to the anomaly score
        y = (1.0 - alpha_expanded) * z + (1.0 + alpha_expanded) * mha_out
        
        return y, alpha