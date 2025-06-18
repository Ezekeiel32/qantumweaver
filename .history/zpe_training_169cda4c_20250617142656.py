import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from datetime import datetime
import json
import requests
from typing import Dict, List, Optional
import joblib
from src.lib.job_logger import JobLogger

# ZPEDeepNet Definition
class ZPEDeepNet(nn.Module):
    def __init__(self, output_size=10, sequence_length=32):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ZPE flows with small random perturbations
        self.zpe_flows = [
            torch.ones(sequence_length, device=self.device) + 
            torch.randn(sequence_length, device=self.device) * 0.01 
            for _ in range(6)
        ]

        # Parameters from job 169cda4c
        self.momentum_params = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
        self.strength_params = [0.35, 0.33, 0.31, 0.6, 0.27, 0.5]
        self.noise_params = [0.3, 0.28, 0.26, 0.35, 0.22, 0.25]
        self.coupling_params = [0.85, 0.82, 0.79, 0.76, 0.73, 0.7]

        # Initialize layers with proper weight initialization
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
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def perturb_zpe_flow(self, data, zpe_idx, feature_size):
        with torch.no_grad():
            # Compute batch statistics safely
            batch_mean = torch.mean(data.detach(), dim=0).view(-1)
            batch_mean = torch.clamp(batch_mean, -10, 10)  # Prevent extreme values
            
            # Reshape and compute perturbation
            divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
            batch_mean_truncated = batch_mean[:divisible_size]
            reshaped = batch_mean_truncated.view(-1, self.sequence_length)
            perturbation = torch.mean(reshaped, dim=0)
            
            # Apply noise with safety checks
            noise = torch.randn_like(perturbation) * self.noise_params[zpe_idx]
            perturbation = perturbation + noise
            perturbation = torch.clamp(perturbation, -10, 10)  # Prevent extreme values
            
            # Apply strength parameter
            perturbation = torch.tanh(perturbation * self.strength_params[zpe_idx])
            
            # Update ZPE flow with momentum
            self.zpe_flows[zpe_idx] = (
                self.momentum_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.momentum_params[zpe_idx]) * (1.0 + perturbation)
            )
            
            # Apply coupling parameter
            self.zpe_flows[zpe_idx] = (
                self.coupling_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.coupling_params[zpe_idx]) * torch.ones_like(self.zpe_flows[zpe_idx])
            )
            
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

class TrainingParameters:
    def __init__(self, 
                 learning_rate: float = 0.0012,
                 weight_decay: float = 0.08,
                 batch_size: int = 32,
                 epochs: int = 60,
                 label_smoothing: float = 0.1,
                 momentum_params: List[float] = None,
                 strength_params: List[float] = None,
                 noise_params: List[float] = None,
                 coupling_params: List[float] = None,
                 model_name: str = "zpe_model"):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_smoothing = label_smoothing
        self.momentum_params = momentum_params or [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
        self.strength_params = strength_params or [0.35, 0.33, 0.31, 0.6, 0.27, 0.5]
        self.noise_params = noise_params or [0.3, 0.28, 0.26, 0.35, 0.22, 0.25]
        self.coupling_params = coupling_params or [0.85, 0.82, 0.79, 0.76, 0.73, 0.7]
        self.model_name = model_name

    def to_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "label_smoothing": self.label_smoothing,
            "momentum_params": self.momentum_params,
            "strength_params": self.strength_params,
            "noise_params": self.noise_params,
            "coupling_params": self.coupling_params,
            "model_name": self.model_name
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingParameters':
        return cls(**data)

    def to_feature_vector(self) -> np.ndarray:
        """Convert parameters to a feature vector for model prediction."""
        features = [
            self.learning_rate,
            self.weight_decay,
            self.batch_size,
            self.epochs,
            self.label_smoothing
        ]
        features.extend(self.momentum_params)
        features.extend(self.strength_params)
        features.extend(self.noise_params)
        features.extend(self.coupling_params)
        return np.array(features).reshape(1, -1)

def get_parameter_suggestions(current_params: TrainingParameters) -> TrainingParameters:
    """Get parameter suggestions using local joblib models."""
    try:
        # Load the models
        val_accuracy_model = joblib.load('val_accuracy_model.joblib')
        zpe_effects_model = joblib.load('zpe_effects_model.joblib')
        
        # Get current features
        features = current_params.to_feature_vector()
        
        # Predict validation accuracy and ZPE effects
        predicted_accuracy = val_accuracy_model.predict(features)[0]
        predicted_zpe_effects = zpe_effects_model.predict(features)[0]
        
        # Adjust parameters based on predictions
        new_params = TrainingParameters(
            learning_rate=current_params.learning_rate * (1.0 + 0.1 * (predicted_accuracy - 95) / 5),
            weight_decay=current_params.weight_decay * (1.0 + 0.1 * (predicted_accuracy - 95) / 5),
            batch_size=current_params.batch_size,
            epochs=current_params.epochs,
            label_smoothing=current_params.label_smoothing,
            momentum_params=[p * (1.0 + 0.1 * e) for p, e in zip(current_params.momentum_params, predicted_zpe_effects)],
            strength_params=[p * (1.0 + 0.1 * e) for p, e in zip(current_params.strength_params, predicted_zpe_effects)],
            noise_params=[p * (1.0 + 0.1 * e) for p, e in zip(current_params.noise_params, predicted_zpe_effects)],
            coupling_params=[p * (1.0 + 0.1 * e) for p, e in zip(current_params.coupling_params, predicted_zpe_effects)],
            model_name=f"{current_params.model_name}_optimized"
        )
        
        print(f"\nPredicted validation accuracy: {predicted_accuracy:.2f}%")
        print(f"Predicted ZPE effects: {predicted_zpe_effects}")
        
        return new_params
    except Exception as e:
        print(f"Warning: Error getting parameter suggestions: {e}")
        return current_params

def train_model(initial_params: Optional[TrainingParameters] = None):
    # Initialize job logger
    logger = JobLogger()
    
    # Use provided parameters or defaults
    params = initial_params or TrainingParameters()
    
    # Generate job ID
    job_id = f"zpe_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Data Setup
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
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=params.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=params.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=2)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ZPEDeepNet(output_size=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=params.epochs)
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    best_val_acc = 0
    zpe_history = []
    
    for epoch in range(params.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{params.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                zpe_effects = model.analyze_zpe_effect()
                print(f'ZPE Effects: {zpe_effects}')
                zpe_history.append({
                    'epoch': epoch + 1,
                    'zpe_effects': zpe_effects.tolist(),
                    'loss': loss.item()
                })
        
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{params.epochs}, Validation Accuracy: {val_acc:.2f}%, Loss: {avg_val_loss:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'params': params.to_dict()
            }, f'{params.model_name}_best.pth')

    # Test
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Accuracy on test set: {accuracy:.2f}%, Loss: {avg_test_loss:.4f}')

    # Get parameter suggestions for next training run
    next_params = get_parameter_suggestions(params)
    print("\nSuggested parameters for next training run:")
    print(json.dumps(next_params.to_dict(), indent=2))

    # Log the job
    job_data = {
        'job_id': job_id,
        'timestamp': datetime.now().isoformat(),
        'model_name': params.model_name,
        'status': 'completed',
        'metrics': {
            'final_accuracy': accuracy,
            'final_loss': avg_test_loss,
            'best_val_accuracy': best_val_acc
        },
        'zpe_effects': model.analyze_zpe_effect().tolist(),
        'parameters': params.to_dict(),
        'zpe_history': zpe_history
    }
    
    logger.log_job(job_data)
    
    return model, accuracy

if __name__ == "__main__":
    # Example usage with default parameters
    model, accuracy = train_model()
    
    # Example usage with custom parameters
    # custom_params = TrainingParameters(
    #     learning_rate=0.001,
    #     weight_decay=0.1,
    #     batch_size=64,
    #     epochs=50,
    #     model_name="zpe_model_custom"
    # )
    # model, accuracy = train_model(custom_params) 