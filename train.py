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

class ZPEQuantumNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, 
                 momentum_params=[0.9]*6, strength_params=[0.5]*6,
                 noise_params=[0.3]*6, coupling_params=[0.8]*6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, 
                         hidden_size if i < 5 else num_classes),
                nn.ReLU() if i < 5 else nn.Identity(),
                ZPELayer(hidden_size if i < 5 else num_classes,
                        momentum_params[i], strength_params[i],
                        noise_params[i], coupling_params[i])
            ) for i in range(6)
        ])
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

def update_job_status(job_id, status, log_message=None, metrics=None):
    job_file = os.path.join('logs.json', f'{job_id}.json')
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        job_data['status'] = status
        if log_message:
            job_data['log_messages'].append(log_message)
        if metrics:
            job_data['metrics'].append(metrics)
            
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)

def train(job_id, params):
    try:
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        update_job_status(job_id, 'running', f'Using device: {device}')

        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batchSize'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['batchSize'])
        
        update_job_status(job_id, 'running', 'Datasets loaded successfully')

        # Initialize model
        model = ZPEQuantumNet(
            momentum_params=params['momentumParams'],
            strength_params=params['strengthParams'],
            noise_params=params['noiseParams'],
            coupling_params=params['couplingParams']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learningRate'], 
                             weight_decay=params['weightDecay'])
        
        update_job_status(job_id, 'running', 'Model initialized')

        # Training loop
        for epoch in range(params['totalEpochs']):
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
                    update_job_status(
                        job_id, 
                        'running',
                        f'Epoch {epoch+1}/{params["totalEpochs"]} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}'
                    )
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            train_accuracy = 100. * correct / total
            val_accuracy = 100. * val_correct / val_total
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_accuracy,
                'val_loss': val_loss / len(test_loader),
                'val_accuracy': val_accuracy
            }
            
            update_job_status(
                job_id,
                'running',
                f'Epoch {epoch+1} completed - Train Acc: {train_accuracy:.2f}% - Val Acc: {val_accuracy:.2f}%',
                metrics
            )
        
        # Save model
        model_path = os.path.join('models', f'{params["modelName"]}.pt')
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        update_job_status(job_id, 'completed', f'Training completed. Model saved to {model_path}')
        
    except Exception as e:
        update_job_status(job_id, 'failed', f'Error during training: {str(e)}')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    
    params = json.loads(args.params)
    train(args.job_id, params) 