"""
Generates simulated multimodal industrial data (images and kinematic sensors)
and injects random single-modality failures for training the MMRCA framework.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class IndustrialMultimodalDataset(Dataset):
    """
    Simulated Dataset for Cyber-Physical Production Systems.
    Generates clean and corrupted pairs for Spatial (Camera) and Kinematic (Sensor) modalities.
    """
    def __init__(self, num_samples: int = 1000, seq_len_sensor: int = 50, img_size: int = 256):
        """
        Args:
            num_samples (int): Total number of samples in the dataset.
            seq_len_sensor (int): Temporal resolution (T_m) of the kinematic sensor.
            img_size (int): Height and width of the camera input.
        """
        self.num_samples = num_samples
        self.seq_len_sensor = seq_len_sensor
        self.img_size = img_size
        
        # Determine token counts based on the CNN architecture in model.py
        # A 256x256 image passed through 5 stride-2 layers becomes 8x8 = 64 tokens.
        self.num_cam_tokens = 64
        self.num_sensor_tokens = seq_len_sensor

    def __len__(self) -> int:
        return self.num_samples

    def _inject_failure(self, x: torch.Tensor, modality_type: str) -> torch.Tensor:
        """Injects signal-dependent multiplicative noise to simulate a sensor failure."""
        corrupted_x = x.clone()
        if modality_type == 'image':
            # Simulate occlusion/noise in the image
            noise = torch.randn_like(corrupted_x) * 2.0
            mask = (torch.rand(*corrupted_x.shape) > 0.5).float()
            corrupted_x = corrupted_x + (noise * mask)
        elif modality_type == 'sensor':
            # Simulate structural gain fluctuations in the kinematic sensor
            gain_fault = torch.rand_like(corrupted_x) * 5.0
            corrupted_x = corrupted_x * (1.0 + gain_fault)
        return corrupted_x

    def __getitem__(self, idx: int):
        # Generate nominal (clean) data
        clean_cam = torch.randn(3, self.img_size, self.img_size) # RGB Image
        clean_sensor = torch.randn(self.seq_len_sensor, 134)     # 134-dimensional kinematic sensor

        corrupted_cam = clean_cam.clone()
        corrupted_sensor = clean_sensor.clone()
        
        # Token-level fault mask for the BCE Detection Loss
        total_tokens = self.num_cam_tokens + self.num_sensor_tokens
        fault_mask = torch.zeros(total_tokens)

        # Random Single-Modality Failure Injection (k ~ U{1, M})
        fault_choice = np.random.choice(['camera', 'sensor', 'clean'])
        
        if fault_choice == 'camera':
            corrupted_cam = self._inject_failure(clean_cam, 'image')
            fault_mask[:self.num_cam_tokens] = 1.0
        elif fault_choice == 'sensor':
            corrupted_sensor = self._inject_failure(clean_sensor, 'sensor')
            fault_mask[self.num_cam_tokens:] = 1.0

        return {
            'clean_cam': clean_cam,
            'clean_sensor': clean_sensor,
            'corrupted_cam': corrupted_cam,
            'corrupted_sensor': corrupted_sensor,
            'fault_mask': fault_mask
        }