import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime

# --- ZPEDeepNet Model ---
class ZPEDeepNet(nn.Module):
    def __init__(self, output_size=10, sequence_length=32, momentum_params=None, strength_params=None, noise_params=None, coupling_params=None):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zpe_flows = [torch.ones(sequence_length, device=self.device) for _ in range(6)]
        self.momentum_params = momentum_params or [0.9]*6
        self.strength_params = strength_params or [0.35]*6
        self.noise_params = noise_params or [0.3]*6
        self.coupling_params = coupling_params or [0.85]*6
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
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )
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
        with torch.no_grad():
            batch_mean = torch.mean(data.detach(), dim=0).view(-1)
            divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
            batch_mean_truncated = batch_mean[:divisible_size]
            reshaped = batch_mean_truncated.view(-1, self.sequence_length)
            perturbation = torch.mean(reshaped, dim=0)
            noise = torch.randn_like(perturbation) * self.noise_params[zpe_idx]
            perturbation = perturbation + noise
            perturbation = torch.tanh(perturbation * self.strength_params[zpe_idx])
            self.zpe_flows[zpe_idx] = (
                self.momentum_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.momentum_params[zpe_idx]) * (1.0 + perturbation)
            )
            self.zpe_flows[zpe_idx] = (
                self.coupling_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.coupling_params[zpe_idx]) * torch.ones_like(self.zpe_flows[zpe_idx])
            )
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

def update_job_status(job_id, status, log_message=None, metrics=None):
    job_file = os.path.join('logs.json', f'{job_id}.json')
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        job_data['status'] = status
        if log_message:
            job_data['log_messages'].append(log_message)
        if metrics:
            if 'metrics' not in job_data:
                job_data['metrics'] = []
            job_data['metrics'].append(metrics)
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    params = json.loads(args.params)
    job_id = args.job_id
    # Hyperparameters
    learning_rate = params.get('learningRate', 0.001)
    weight_decay = params.get('weightDecay', 0.0001)
    batch_size = params.get('batchSize', 32)
    epochs = params.get('totalEpochs', 30)
    label_smoothing = params.get('labelSmoothing', 0.1)
    momentum_params = params.get('momentumParams', [0.9]*6)
    strength_params = params.get('strengthParams', [0.35]*6)
    noise_params = params.get('noiseParams', [0.3]*6)
    coupling_params = params.get('couplingParams', [0.85]*6)
    model_name = params.get('modelName', 'zpe_model')
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
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_job_status(job_id, 'running', f'Using device: {device}')
    model = ZPEDeepNet(output_size=10, sequence_length=32, momentum_params=momentum_params, strength_params=strength_params, noise_params=noise_params, coupling_params=coupling_params).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        # Training loop
    for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                if batch_idx % 100 == 0:
                update_job_status(job_id, 'running', f'Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
        scheduler.step()
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
            for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            train_accuracy = 100. * correct / total
            val_accuracy = 100. * val_correct / val_total
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_accuracy,
            'val_loss': val_loss / len(val_loader),
                'val_accuracy': val_accuracy
            }
        update_job_status(job_id, 'running', f'Epoch {epoch+1} completed - Train Acc: {train_accuracy:.2f}% - Val Acc: {val_accuracy:.2f}%', metrics)
        # Save model
        os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', f'{model_name}_best.pth'))
    update_job_status(job_id, 'completed', f'Training completed. Model saved to models/{model_name}_best.pth')

if __name__ == "__main__":
    main() 