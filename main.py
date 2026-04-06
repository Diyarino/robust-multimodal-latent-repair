"""
Training and evaluation loops for the MMRCA framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import IndustrialMultimodalDataset
from model import MMRCA

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Hyperparameters from paper
    batch_size = 16
    epochs = 200
    learning_rate = 1e-4
    gamma = 0.01 # Regulates sensitivity of detection mechanism
    
    # Initialize Dataset & DataLoader
    train_dataset = IndustrialMultimodalDataset(num_samples=1000)
    test_dataset = IndustrialMultimodalDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model, Optimizer, and Loss Functions
    model = MMRCA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_recon_loss = 0.0
        train_bce_loss = 0.0
        
        for batch in train_loader:
            clean_cam = batch['clean_cam'].to(device)
            clean_sensor = batch['clean_sensor'].to(device)
            corr_cam = batch['corrupted_cam'].to(device)
            corr_sensor = batch['corrupted_sensor'].to(device)
            fault_mask = batch['fault_mask'].to(device) # Shape: (B, T_total)

            optimizer.zero_grad()

            # Forward Pass: Feed corrupted data
            hat_cam, hat_sensor, alpha = model(corr_cam, corr_sensor)
            
            # Optimization Objective (Eq. 12)
            # Reconstruct towards the CLEAN target signals
            loss_cam = criterion_mse(hat_cam, clean_cam)
            loss_sensor = criterion_mse(hat_sensor, clean_sensor)
            recon_loss = loss_cam + loss_sensor
            
            det_loss = criterion_bce(alpha, fault_mask)
            
            total_loss = recon_loss + gamma * det_loss

            # Backward pass & update
            total_loss.backward()
            optimizer.step()

            train_recon_loss += recon_loss.item()
            train_bce_loss += det_loss.item()

        # --- Evaluation Loop ---
        model.eval()
        test_recon_loss = 0.0
        test_det_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                clean_cam = batch['clean_cam'].to(device)
                clean_sensor = batch['clean_sensor'].to(device)
                corr_cam = batch['corrupted_cam'].to(device)
                corr_sensor = batch['corrupted_sensor'].to(device)
                fault_mask = batch['fault_mask'].to(device)

                hat_cam, hat_sensor, alpha = model(corr_cam, corr_sensor)
                
                loss_cam = criterion_mse(hat_cam, clean_cam)
                loss_sensor = criterion_mse(hat_sensor, clean_sensor)
                
                test_recon_loss += (loss_cam + loss_sensor).item()
                test_det_loss += criterion_bce(alpha, fault_mask).item()

        # Print Metrics every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            avg_tr_recon = train_recon_loss / len(train_loader)
            avg_te_recon = test_recon_loss / len(test_loader)
            print(f"Epoch [{epoch:03d}/{epochs}] "
                  f"| Train Recon Loss: {avg_tr_recon:.4f} "
                  f"| Test Recon Loss: {avg_te_recon:.4f} "
                  f"| Test BCE: {test_det_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    main()