import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from datetime import datetime

def calculate_quantum_circuit_size():
    # 2^5 = 32
    base_size = 2 ** 5
    # 32 * 11/16 = 22
    adjusted_size = base_size * (11/16)
    # 22 + 3.33 * 3 = 32
    final_size = adjusted_size + (3.33 * 3)
    # Add quantum precision factor
    quantum_precision = 0.0000001
    return int(final_size), quantum_precision

# ZPEDeepNet Definition
class ZPEDeepNet(nn.Module):
    def __init__(self):
        super(ZPEDeepNet, self).__init__()
        # Calculate quantum circuit size using the formula
        self.sequence_length, self.quantum_precision = calculate_quantum_circuit_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ZPE flows with quantum precision
        self.zpe_flows = [
            torch.ones(self.sequence_length, device=self.device) + 
            torch.randn(self.sequence_length, device=self.device) * self.quantum_precision 
            for _ in range(6)
        ]
        
        # Momentum and strength parameters from job 169cda4c
        self.momentum_params = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
        self.strength_params = [0.35, 0.33, 0.31, 0.6, 0.27, 0.5]
        self.noise_params = [0.3, 0.28, 0.26, 0.35, 0.22, 0.25]
        self.coupling_params = [0.85, 0.82, 0.79, 0.76, 0.73, 0.7]

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

        # Residual connections
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )

    def perturb_zpe_flow(self, data, zpe_idx, feature_size):
        batch_mean = torch.mean(data.detach(), dim=0).view(-1)
        divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
        batch_mean_truncated = batch_mean[:divisible_size]
        reshaped = batch_mean_truncated.view(-1, self.sequence_length)
        perturbation = torch.mean(reshaped, dim=0)
        
        # Apply quantum precision to perturbation
        perturbation = perturbation + torch.randn_like(perturbation) * self.quantum_precision
        
        # Apply noise parameter
        noise = torch.randn_like(perturbation) * self.noise_params[zpe_idx]
        perturbation = perturbation + noise
        
        # Apply strength parameter
        perturbation = torch.tanh(perturbation * self.strength_params[zpe_idx])
        
        # Update ZPE flow with momentum and quantum precision
        with torch.no_grad():
            self.zpe_flows[zpe_idx] = (
                self.momentum_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.momentum_params[zpe_idx]) * (1.0 + perturbation)
            )
            # Apply coupling parameter
            self.zpe_flows[zpe_idx] = (
                self.coupling_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.coupling_params[zpe_idx]) * torch.ones_like(self.zpe_flows[zpe_idx])
            )
            # Add quantum precision to the flow
            self.zpe_flows[zpe_idx] += torch.randn_like(self.zpe_flows[zpe_idx]) * self.quantum_precision
            # Clamp values
            self.zpe_flows[zpe_idx] = torch.clamp(self.zpe_flows[zpe_idx], 0.8, 1.2)

    def apply_zpe(self, x, zpe_idx, spatial=True):
        self.perturb_zpe_flow(x, zpe_idx, x.size(1) if spatial else x.size(-1))
        flow = self.zpe_flows[zpe_idx]
        if spatial:
            size = x.size(2) * x.size(3)
            flow_expanded = flow.repeat(size // self.sequence_length + 1)[:size].view(1, 1, x.size(2), x.size(3))
            flow_expanded = flow_expanded.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        else:
            flow_expanded = flow.repeat(x.size(-1) // self.sequence_length + 1)[:x.size(-1)].view(1, -1)
            flow_expanded = flow_expanded.expand(x.size(0), x.size(-1))
        return x * flow_expanded

    def forward(self, x):
        x = self.apply_zpe(x, 0)
        residual = self.shortcut1(x)
        x = self.conv1(x) + residual
        
        x = self.apply_zpe(x, 1)
        residual = self.shortcut2(x)
        x = self.conv2(x) + residual
        
        x = self.apply_zpe(x, 2)
        residual = self.shortcut3(x)
        x = self.conv3(x) + residual
        
        x = self.apply_zpe(x, 3)
        residual = self.shortcut4(x)
        x = self.conv4(x) + residual
        
        x = self.apply_zpe(x, 4)
        x = self.fc(x)
        x = self.apply_zpe(x, 5, spatial=False)
        return x

    def analyze_zpe_effect(self):
        return [torch.mean(torch.abs(flow - 1.0)).item() for flow in self.zpe_flows]

def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    # Split training data into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize model
    model = ZPEDeepNet().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing from job 169cda4c
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0012,  # Learning rate from job 169cda4c
        weight_decay=0.08  # Weight decay from job 169cda4c
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=60)  # 60 epochs from job 169cda4c

    # Training loop
    best_val_acc = 0.0
    for epoch in range(60):  # 60 epochs from job 169cda4c
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Get ZPE effects
        zpe_effects = model.analyze_zpe_effect()
        
        # Print epoch results
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'ZPE Effects: {[f"{x:.4f}" for x in zpe_effects]}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'zpe_effects': zpe_effects
            }, f'best_model_169cda4c.pth')
        
        # Update learning rate
        scheduler.step()

    # Test phase
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f'\nFinal Test Results:')
    print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    train_model() 