import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List

class TrainingLogger:
    def __init__(self, csv_path: str = "training_logs.csv", logs_dir: str = "logs.json"):
        self.csv_path = csv_path
        self.logs_dir = logs_dir
        self._ensure_csv_exists()
        self._process_historical_logs()

    def _ensure_csv_exists(self):
        """Ensure the CSV file exists with the correct headers."""
        if not os.path.exists(self.csv_path):
            headers = [
                'Job_ID', 'Epoch', 'Total_Epochs', 'Timestamp', 'Model_Name', 'Base_Config_ID',
                'Train_Loss', 'Val_Loss', 'Val_Accuracy', 'Epoch_Time', 'Conv1_Channels',
                'Conv2_Channels', 'Conv3_Channels', 'Conv4_Channels', 'FC_Hidden', 'Dropout_FC',
                'Energy_Scale', 'Learning_Rate', 'Mixup_Alpha', 'Batch_Size', 'Momentum_Factor',
                'Label_Smoothing', 'Quantum_Circuit_Size', 'Quantum_Mode', 'Momentum_Params',
                'Strength_Params', 'Noise_Params', 'Coupling_Params', 'Cycle_Scale',
                'zpe_effects', 'Weight_Decay'
            ]
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _process_historical_logs(self):
        """Process all historical log files and add them to the CSV if not already present."""
        if not os.path.exists(self.logs_dir):
            return

        # Get existing job IDs from CSV
        existing_jobs = set()
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_jobs = {row['Job_ID'] for row in reader}

        # Process each log file
        for filename in os.listdir(self.logs_dir):
            if not filename.endswith('.json'):
                continue

            job_id = filename.replace('.json', '')
            if job_id in existing_jobs:
                continue

            try:
                with open(os.path.join(self.logs_dir, filename), 'r') as f:
                    job_data = json.load(f)

                if job_data.get('status') != 'completed':
                    continue

                # Process each epoch from log messages
                for log_msg in job_data.get('log_messages', []):
                    epoch_match = log_msg.match(r'E(\d+)\s+END\s+-\s+TrainL:\s+([\d.]+),\s+ValAcc:\s+([\d.]+)%,\s+ValL:\s+([\d.]+)')
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        train_loss = float(epoch_match.group(2))
                        val_accuracy = float(epoch_match.group(3))
                        val_loss = float(epoch_match.group(4))

                        # Find corresponding ZPE effects
                        zpe_effects = None
                        for zpe_msg in job_data.get('log_messages', []):
                            if f'E{epoch} END' in zpe_msg and 'ZPE:' in zpe_msg:
                                zpe_match = zpe_msg.match(r'ZPE:\s+\[([\d.,\s]+)\]')
                                if zpe_match:
                                    zpe_effects = [float(x.strip()) for x in zpe_match.group(1).split(',')]
                                break

                        if zpe_effects is None:
                            zpe_effects = job_data.get('zpe_effects', [0.0] * 6)

                        epoch_data = {
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'epoch_time': 0.0  # Historical data doesn't have timing
                        }

                        self.log_epoch(job_data, epoch_data)

            except Exception as e:
                print(f"Error processing historical log {filename}: {str(e)}")

    def log_epoch(self, job_data: Dict[str, Any], epoch_data: Dict[str, Any]):
        """Log a single epoch's data to the CSV file."""
        params = job_data.get('parameters', {})
        
        # Extract ZPE effects
        zpe_effects = job_data.get('zpe_effects', [])
        if len(zpe_effects) < 6:
            zpe_effects = list(zpe_effects) + [0.0] * (6 - len(zpe_effects))
        
        # Extract momentum, strength, noise, and coupling parameters
        momentum_params = params.get('momentumParams', [0.0] * 6)
        strength_params = params.get('strengthParams', [0.0] * 6)
        noise_params = params.get('noiseParams', [0.0] * 6)
        coupling_params = params.get('couplingParams', [0.0] * 6)
        
        # Ensure all parameter arrays are of length 6
        for param_list in [momentum_params, strength_params, noise_params, coupling_params]:
            if len(param_list) < 6:
                param_list.extend([0.0] * (6 - len(param_list)))

        # Prepare the row data
        row = [
            job_data.get('job_id', ''),
            epoch_data.get('epoch', 0),
            params.get('totalEpochs', 0),
            datetime.now().isoformat(),
            params.get('modelName', ''),
            params.get('baseConfigId', ''),
            epoch_data.get('train_loss', 0.0),
            epoch_data.get('val_loss', 0.0),
            epoch_data.get('val_accuracy', 0.0),
            epoch_data.get('epoch_time', 0.0),
            params.get('conv1Channels', 0),
            params.get('conv2Channels', 0),
            params.get('conv3Channels', 0),
            params.get('conv4Channels', 0),
            params.get('fcHidden', 0),
            params.get('dropoutFC', 0.0),
            params.get('energyScale', 1.0),
            params.get('learningRate', 0.0),
            params.get('mixupAlpha', 0.0),
            params.get('batchSize', 0),
            params.get('momentumFactor', 0.0),
            params.get('labelSmoothing', 0.0),
            params.get('quantumCircuitSize', 0),
            params.get('quantumMode', False),
            ','.join(map(str, momentum_params)),
            ','.join(map(str, strength_params)),
            ','.join(map(str, noise_params)),
            ','.join(map(str, coupling_params)),
            params.get('cycleScale', 1.0),
            ','.join(map(str, zpe_effects)),
            params.get('weightDecay', 0.0)
        ]

        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_job_completion(self, job_data: Dict[str, Any]):
        """Log the final job completion data."""
        # Extract the final epoch data
        final_epoch = {
            'epoch': job_data.get('current_epoch', 0),
            'train_loss': job_data.get('loss', 0.0),
            'val_loss': job_data.get('loss', 0.0),
            'val_accuracy': job_data.get('accuracy', 0.0),
            'epoch_time': 0.0  # You might want to calculate this from start/end times
        }
        
        self.log_epoch(job_data, final_epoch) 