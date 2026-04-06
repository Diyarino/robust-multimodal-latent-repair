"""
Unified Multimodal Autoencoder integrated with the RCA mechanism.
"""

import torch
import torch.nn as nn
from rca import ResidualContrastiveAttention

class MMRCA(nn.Module):
    """
    Robust Multimodal Latent Repair Framework.
    """
    def __init__(self, latent_dim: int = 288, sensor_seq_len: int = 50):
        super(MMRCA, self).__init__()
        self.latent_dim = latent_dim
        self.num_cam_tokens = 64 # From 256x256 -> 8x8 spatial grid
        
        # --- Modality-Specific Encoders ---
        # Camera Encoder (5-layer CNN)
        self.cam_encoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2)
        )
        self.cam_proj = nn.Linear(8, latent_dim)
        
        # Sensor Encoder (5-layer MLP applied per timestep)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(134, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 392), nn.LeakyReLU(0.2)
        )
        self.sensor_proj = nn.Linear(392, latent_dim)

        # --- Positional Encodings ---
        self.cam_pe = nn.Parameter(torch.randn(1, self.num_cam_tokens, latent_dim))
        self.sensor_pe = nn.Parameter(torch.randn(1, sensor_seq_len, latent_dim))

        # --- Residual-Contrastive Attention Module ---
        self.rca = ResidualContrastiveAttention(embed_dim=latent_dim, num_heads=8)

        # --- Modality-Specific Decoders ---
        self.cam_inv_proj = nn.Linear(latent_dim, 8)
        self.cam_decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.sensor_inv_proj = nn.Linear(latent_dim, 392)
        self.sensor_decoder = nn.Sequential(
            nn.Linear(392, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 134)
        )

    def forward(self, cam_in, sensor_in):
        B = cam_in.size(0)
        
        # 1. Encoding
        # Camera: (B, 3, 256, 256) -> (B, 8, 8, 8)
        z_cam = self.cam_encoder(cam_in)
        z_cam = z_cam.view(B, 8, -1).transpose(1, 2) # Flatten spatial to (B, 64, 8)
        z_cam = self.cam_proj(z_cam) + self.cam_pe  # (B, 64, D)
        
        # Sensor: (B, T, 134) -> (B, T, D)
        z_sensor = self.sensor_encoder(sensor_in)
        z_sensor = self.sensor_proj(z_sensor) + self.sensor_pe

        # 2. Unified State Integration
        z_raw = torch.cat([z_cam, z_sensor], dim=1) # (B, T_total, D)

        # 3. RCA Intervention
        y_repaired, alpha = self.rca(z_raw)

        # 4. Decoding
        y_cam = y_repaired[:, :self.num_cam_tokens, :]
        y_sensor = y_repaired[:, self.num_cam_tokens:, :]

        # Decode Camera
        y_cam = self.cam_inv_proj(y_cam).transpose(1, 2).view(B, 8, 8, 8)
        hat_cam = self.cam_decoder(y_cam)

        # Decode Sensor
        y_sensor = self.sensor_inv_proj(y_sensor)
        hat_sensor = self.sensor_decoder(y_sensor)

        return hat_cam, hat_sensor, alpha