import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import csv
from datetime import datetime
import os

# --- Parameter Generators ---
def generate_zpe_flow_params(cycle_length=32):
    # Optimized ranges based on high-accuracy patterns
    momentum = random.uniform(0.75, 0.95)  # Higher momentum for stability
    strength = random.uniform(0.15, 0.35)  # Moderate strength for controlled effects
    noise = random.uniform(0.1, 0.25)      # Lower noise for stability
    coupling = random.uniform(0.65, 0.85)  # Balanced coupling
    
    # Quantum cycle modulation
    cycle_factor = (11/16) * (cycle_length / 32)
    false_ten = 3.33 * 3
    strength *= cycle_factor * (false_ten / 10)
    
    # Ensure parameters are within optimal ranges
    momentum = max(0.75, min(0.95, momentum))
    strength = max(0.15, min(0.35, strength))
    noise = max(0.1, min(0.25, noise))
    coupling = max(0.65, min(0.85, coupling))
    
    return momentum, strength, noise, coupling

def generate_layer_config(base_channels):
    # More stable channel variations
    variation = random.uniform(0.85, 1.15)
    return int(base_channels * variation)

def generate_fc_config(base_size):
    # More stable FC layer variations
    variation = random.uniform(0.8, 1.2)
    return int(base_size * variation)

def generate_augmentation_params():
    # Optimized augmentation parameters for stability
    rotation = random.randint(15, 25)  # Reduced rotation range
    translate = random.uniform(0.15, 0.25)  # More controlled translation
    crop_padding = random.randint(3, 5)  # Reduced padding variation
    erase_prob = random.uniform(0.4, 0.6)  # More balanced erasing
    erase_scale = (random.uniform(0.02, 0.04), random.uniform(0.15, 0.25))  # Controlled erasing
    return rotation, translate, crop_padding, erase_prob, erase_scale

def generate_optimizer_config():
    # Optimized optimizer parameters
    optimizer_type = random.choice(['Adam', 'AdamW'])  # Removed SGD for stability
    lr = random.uniform(0.0008, 0.0015)  # Narrower learning rate range
    weight_decay = random.uniform(2e-5, 5e-4)  # Adjusted weight decay
    scheduler_type = 'CosineAnnealingLR'  # Fixed to most stable scheduler
    t_max = random.randint(25, 35)  # Adjusted epoch range
    return optimizer_type, lr, weight_decay, scheduler_type, t_max

# --- Model Definition ---
class ZPEDeepNet(nn.Module):
    def __init__(self, conv_channels, fc_sizes, sequence_length, dropout_rate, strength, momentum):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.cycle_length = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zpe_flows = [torch.ones(sequence_length, device=self.device) for _ in range(6)]
        self.strength = strength
        self.momentum = momentum
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[3]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[3], fc_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_sizes[1], 10)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )

    def perturb_zpe_flow(self, data, zpe_idx, feature_size):
        batch_mean = torch.mean(data.detach(), dim=0).view(-1)
        divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
        batch_mean_truncated = batch_mean[:divisible_size]
        reshaped = batch_mean_truncated.view(-1, self.sequence_length)
        perturbation = torch.mean(reshaped, dim=0)
        perturbation = torch.tanh(perturbation * self.strength)
        with torch.no_grad():
            self.zpe_flows[zpe_idx] = self.momentum * self.zpe_flows[zpe_idx] + (1 - self.momentum) * (1.0 + perturbation)
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

# --- Logging Function ---
def log_job_to_csv(csv_path, job_id, params, zpe_effects, val_acc):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['job_id', 'momentum', 'strength', 'noise', 'coupling', 'conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'dropout', 'batch_size', 'lr', 'weight_decay', 'num_epochs', 'zpe1', 'zpe2', 'zpe3', 'zpe4', 'zpe5', 'zpe6', 'val_acc'])
        writer.writerow([job_id] + list(params) + list(zpe_effects) + [val_acc])

# --- Main Loop ---
def main(num_jobs=500):
    csv_path = 'zpe_supervised_dataset.csv'
    for job_num in range(num_jobs):
        # Generate optimized parameters
        momentum, strength, noise, coupling = generate_zpe_flow_params()
        conv_channels = [generate_layer_config(c) for c in [64, 128, 256, 512]]
        fc_sizes = [generate_layer_config(s) for s in [2048, 512]]
        dropout_rate = random.uniform(0.35, 0.55)  # Adjusted dropout range
        batch_size = random.choice([32, 64])  # Removed smaller batch sizes
        lr = random.uniform(0.0008, 0.0015)  # Narrower learning rate range
        weight_decay = random.uniform(2e-5, 5e-4)  # Adjusted weight decay
        num_epochs = random.randint(8, 15)  # Increased epochs for better convergence
        sequence_length = random.choice([10, 12])  # Removed smaller sequence lengths

        # Data transforms
        rotation, translate, crop_padding, erase_prob, erase_scale = generate_augmentation_params()
        train_transform = transforms.Compose([
            transforms.RandomRotation(rotation),
            transforms.RandomAffine(degrees=0, translate=(translate, translate)),
            transforms.RandomCrop(28, padding=crop_padding),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=erase_prob, scale=erase_scale)
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

        # Model, optimizer, loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ZPEDeepNet(conv_channels, fc_sizes, sequence_length, dropout_rate, strength, momentum).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0

        # ZPE effects
        zpe_effects = model.analyze_zpe_effect()

        # Log to CSV
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_num}"
        params = [momentum, strength, noise, coupling] + conv_channels + fc_sizes + [dropout_rate, batch_size, lr, weight_decay, num_epochs]
        log_job_to_csv(csv_path, job_id, params, zpe_effects, val_acc)
        print(f"Logged job {job_id} with val_acc={val_acc:.2f}")

if __name__ == "__main__":
    main(num_jobs=500)  # Generate 500 jobs for a manageable test run
