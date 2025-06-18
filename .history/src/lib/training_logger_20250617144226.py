import csv
import time
from datetime import datetime
from typing import Dict, List, Any
import os

class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_job_id = None
        self.current_file = None
        self.writer = None
        self.start_time = None

    def start_job(self, job_id: str, model_name: str, base_config_id: str, params: Dict[str, Any]):
        """Start logging a new training job."""
        self.current_job_id = job_id
        self.start_time = time.time()
        
        # Create CSV file for this job
        filename = f"{job_id}.csv"
        filepath = os.path.join(self.log_dir, filename)
        self.current_file = open(filepath, 'w', newline='')
        
        # Initialize CSV writer with headers
        fieldnames = [
            'Job_ID', 'Epoch', 'Total_Epochs', 'Timestamp', 'Model_Name', 'Base_Config_ID',
            'Train_Loss', 'Val_Loss', 'Val_Accuracy', 'Epoch_Time', 'Conv1_Channels',
            'Conv2_Channels', 'Conv3_Channels', 'Conv4_Channels', 'FC_Hidden', 'Dropout_FC',
            'Energy_Scale', 'Learning_Rate', 'Mixup_Alpha', 'Batch_Size', 'Momentum_Factor',
            'Label_Smoothing', 'Quantum_Circuit_Size', 'Quantum_Mode', 'Momentum_Params',
            'Strength_Params', 'Noise_Params', 'Coupling_Params', 'Cycle_Scale',
            'zpe_effects', 'Weight_Decay'
        ]
        self.writer = csv.DictWriter(self.current_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log_epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, Any], params: Dict[str, Any], zpe_effects: List[float]):
        """Log a single epoch's data."""
        if not self.writer:
            raise RuntimeError("No active job. Call start_job first.")

        epoch_start_time = time.time()
        
        # Format the data according to the CSV structure
        row = {
            'Job_ID': self.current_job_id,
            'Epoch': epoch,
            'Total_Epochs': total_epochs,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Model_Name': params.get('modelName', ''),
            'Base_Config_ID': params.get('baseConfigId', ''),
            'Train_Loss': metrics.get('train_loss', 0.0),
            'Val_Loss': metrics.get('val_loss', 0.0),
            'Val_Accuracy': metrics.get('val_accuracy', 0.0),
            'Epoch_Time': time.time() - epoch_start_time,
            'Conv1_Channels': params.get('conv1Channels', 0),
            'Conv2_Channels': params.get('conv2Channels', 0),
            'Conv3_Channels': params.get('conv3Channels', 0),
            'Conv4_Channels': params.get('conv4Channels', 0),
            'FC_Hidden': params.get('fcHidden', 0),
            'Dropout_FC': params.get('dropoutFC', 0.0),
            'Energy_Scale': params.get('energyScale', 1.0),
            'Learning_Rate': params.get('learningRate', 0.0),
            'Mixup_Alpha': params.get('mixupAlpha', 0.0),
            'Batch_Size': params.get('batchSize', 0),
            'Momentum_Factor': params.get('momentumFactor', 0.0),
            'Label_Smoothing': params.get('labelSmoothing', 0.0),
            'Quantum_Circuit_Size': params.get('quantumCircuitSize', 0),
            'Quantum_Mode': int(params.get('quantumMode', False)),
            'Momentum_Params': str(params.get('momentumParams', [])),
            'Strength_Params': str(params.get('strengthParams', [])),
            'Noise_Params': str(params.get('noiseParams', [])),
            'Coupling_Params': str(params.get('couplingParams', [])),
            'Cycle_Scale': params.get('cycleScale', 0),
            'zpe_effects': str(zpe_effects),
            'Weight_Decay': params.get('weightDecay', 0.0)
        }
        
        self.writer.writerow(row)
        self.current_file.flush()

    def end_job(self):
        """End the current training job logging."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
            self.writer = None
            self.current_job_id = None
            self.start_time = None

    def __del__(self):
        """Ensure file is closed when object is destroyed."""
        if self.current_file:
            self.current_file.close() 