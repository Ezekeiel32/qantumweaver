import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

class JobLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.csv_file = os.path.join(log_dir, "training_logs.csv")
        self._ensure_log_dir()
        self._ensure_csv_header()

    def _ensure_log_dir(self):
        """Ensure the log directory exists."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _ensure_csv_header(self):
        """Ensure the CSV file exists with proper headers."""
        if not os.path.exists(self.csv_file):
            headers = [
                "Job_ID", "Epoch", "Total_Epochs", "Timestamp", "Model_Name",
                "Base_Config_ID", "Train_Loss", "Val_Loss", "Val_Accuracy",
                "Epoch_Time", "Conv1_Channels", "Conv2_Channels", "Conv3_Channels",
                "Conv4_Channels", "FC_Hidden", "Dropout_FC", "Energy_Scale",
                "Learning_Rate", "Mixup_Alpha", "Batch_Size", "Momentum_Factor",
                "Label_Smoothing", "Quantum_Circuit_Size", "Quantum_Mode",
                "Momentum_Params", "Strength_Params", "Noise_Params",
                "Coupling_Params", "Cycle_Scale", "zpe_effects", "Weight_Decay"
            ]
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_epoch(self, 
                 job_id: str,
                 epoch: int,
                 total_epochs: int,
                 model_name: str,
                 base_config_id: Optional[str],
                 train_loss: float,
                 val_loss: float,
                 val_accuracy: float,
                 epoch_time: float,
                 conv_channels: List[int],
                 fc_hidden: int,
                 dropout_fc: float,
                 energy_scale: float,
                 learning_rate: float,
                 mixup_alpha: float,
                 batch_size: int,
                 momentum_factor: float,
                 label_smoothing: float,
                 quantum_circuit_size: int,
                 quantum_mode: bool,
                 momentum_params: List[float],
                 strength_params: List[float],
                 noise_params: List[float],
                 coupling_params: List[float],
                 cycle_scale: float,
                 zpe_effects: List[float],
                 weight_decay: float):
        """Log a single epoch's data to the CSV file."""
        
        # Ensure conv_channels has 4 values
        conv_channels = (conv_channels + [0] * 4)[:4]
        
        # Convert lists to JSON strings
        momentum_params_str = json.dumps(momentum_params)
        strength_params_str = json.dumps(strength_params)
        noise_params_str = json.dumps(noise_params)
        coupling_params_str = json.dumps(coupling_params)
        zpe_effects_str = json.dumps(zpe_effects)

        row = [
            job_id,
            epoch,
            total_epochs,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            base_config_id or "",
            train_loss,
            val_loss,
            val_accuracy,
            epoch_time,
            conv_channels[0],
            conv_channels[1],
            conv_channels[2],
            conv_channels[3],
            fc_hidden,
            dropout_fc,
            energy_scale,
            learning_rate,
            mixup_alpha,
            batch_size,
            momentum_factor,
            label_smoothing,
            quantum_circuit_size,
            1 if quantum_mode else 0,
            momentum_params_str,
            strength_params_str,
            noise_params_str,
            coupling_params_str,
            cycle_scale,
            zpe_effects_str,
            weight_decay
        ]

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_job_completion(self, job_data: Dict[str, Any]):
        """Log a completed job's data to the CSV file."""
        # Extract parameters from job_data
        params = job_data.get('parameters', {})
        
        # Log each epoch's data
        for epoch_data in job_data.get('epoch_history', []):
            self.log_epoch(
                job_id=job_data['job_id'],
                epoch=epoch_data['epoch'],
                total_epochs=params.get('totalEpochs', 0),
                model_name=params.get('modelName', ''),
                base_config_id=params.get('baseConfigId'),
                train_loss=epoch_data.get('train_loss', 0.0),
                val_loss=epoch_data.get('val_loss', 0.0),
                val_accuracy=epoch_data.get('val_accuracy', 0.0),
                epoch_time=epoch_data.get('epoch_time', 0.0),
                conv_channels=params.get('convChannels', [0, 0, 0, 0]),
                fc_hidden=params.get('fcHidden', 0),
                dropout_fc=params.get('dropoutFC', 0.0),
                energy_scale=params.get('energyScale', 1.0),
                learning_rate=params.get('learningRate', 0.0),
                mixup_alpha=params.get('mixupAlpha', 0.0),
                batch_size=params.get('batchSize', 0),
                momentum_factor=params.get('momentumFactor', 0.0),
                label_smoothing=params.get('labelSmoothing', 0.0),
                quantum_circuit_size=params.get('quantumCircuitSize', 0),
                quantum_mode=params.get('quantumMode', False),
                momentum_params=params.get('momentumParams', [0.0] * 6),
                strength_params=params.get('strengthParams', [0.0] * 6),
                noise_params=params.get('noiseParams', [0.0] * 6),
                coupling_params=params.get('couplingParams', [0.0] * 6),
                cycle_scale=params.get('cycleScale', 0.0),
                zpe_effects=epoch_data.get('zpe_effects', [0.0] * 6),
                weight_decay=params.get('weightDecay', 0.0)
            ) 