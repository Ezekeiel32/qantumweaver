import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class ZPELayer(nn.Module):
    def __init__(self, size, momentum=0.9, strength=0.5, noise=0.3, coupling=0.8):
        super().__init__()
        self.size = size
        self.momentum = momentum
        self.strength = strength
        self.noise = noise
        self.coupling = coupling
        self.register_buffer('velocity', torch.zeros(size))
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(self.velocity) * self.noise
            self.velocity = self.momentum * self.velocity + self.strength * noise
            return x + self.coupling * self.velocity
        return x

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, quantum_circuit_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.zpe1 = ZPELayer(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.zpe2 = ZPELayer(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.zpe3 = ZPELayer(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.zpe4 = ZPELayer(512)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.zpe5 = ZPELayer(1024)
        self.fc2 = nn.Linear(1024, quantum_circuit_size)
        self.zpe6 = ZPELayer(quantum_circuit_size)
        self.fc3 = nn.Linear(quantum_circuit_size, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.zpe1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.zpe2(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.zpe3(x)
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.zpe4(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.zpe5(x)
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.zpe6(x)
        x = self.fc3(x)
        return x

def train_model(params, job_id):
    # Set up logging
    log_file = os.path.join('logs.json', f'{job_id}.json')
    
    # Load and update job data
    with open(log_file, 'r') as f:
        job_data = json.load(f)
    job_data['status'] = 'running'
    with open(log_file, 'w') as f:
        json.dump(job_data, f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data/MNIST', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batchSize'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batchSize'])
    
    # Initialize model
    model = QuantumNeuralNetwork(params['quantumCircuitSize']).to(device)
    
    # Set up ZPE parameters
    for i, layer in enumerate([model.zpe1, model.zpe2, model.zpe3, model.zpe4, model.zpe5, model.zpe6]):
        layer.momentum = params['momentumParams'][i]
        layer.strength = params['strengthParams'][i]
        layer.noise = params['noiseParams'][i]
        layer.coupling = params['couplingParams'][i]
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learningRate'], weight_decay=params['weightDecay'])
    
    # Training loop
    for epoch in range(params['totalEpochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
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
                log_msg = f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' \
                         f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
                job_data['log_messages'].append(log_msg)
                with open(log_file, 'w') as f:
                    json.dump(job_data, f, indent=2)
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        zpe_effects = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Calculate ZPE effects
                for layer in [model.zpe1, model.zpe2, model.zpe3, model.zpe4, model.zpe5, model.zpe6]:
                    zpe_effects.append(float(layer.velocity.abs().mean().cpu()))
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        
        # Log epoch results
        epoch_time = time.time() - start_time
        log_msg = f'END Epoch: {epoch}\tTest loss: {test_loss:.6f}\tAccuracy: {accuracy:.2f}%'
        job_data['log_messages'].append(log_msg)
        job_data['metrics'].append({
            'epoch': epoch,
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy,
            'val_loss': test_loss,
            'val_accuracy': accuracy,
            'zpe_effects': zpe_effects,
            'epoch_time': epoch_time
        })
        
        with open(log_file, 'w') as f:
            json.dump(job_data, f, indent=2)
    
    # Update final job status
    job_data['status'] = 'completed'
    job_data['accuracy'] = accuracy
    job_data['loss'] = test_loss
    with open(log_file, 'w') as f:
        json.dump(job_data, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    
    params = json.loads(args.params)
    train_model(params, args.job_id) 