import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class TrainingLogger:
    def __init__(self, csv_path: str = "training_logs.csv"):
        """Initialize the training logger with the path to the CSV file."""
        self.csv_path = csv_path
        self._ensure_csv_exists()
        self._processed_jobs = set()
        self._process_historical_logs()

    def _ensure_csv_exists(self):
        """Ensure the CSV file exists with proper headers."""
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
        """Process all historical logs from logs.json directory."""
        logs_dir = "logs.json"
        if not os.path.exists(logs_dir):
            return

        # Get existing job IDs from CSV
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            self._processed_jobs = {row['Job_ID'] for row in reader}

        # Process each log file
        for filename in os.listdir(logs_dir):
            if not filename.endswith('.json'):
                continue

            job_id = filename.replace('.json', '')
            if job_id in self._processed_jobs:
                continue

            try:
                with open(os.path.join(logs_dir, filename), 'r') as f:
                    job_data = json.load(f)

                if job_data.get('status') != 'completed':
                    continue

                params = job_data.get('parameters', {})
                for log_msg in job_data.get('log_messages', []):
                    if 'E' in log_msg and 'END' in log_msg:
                        # Extract epoch data
                        epoch_data = self._extract_epoch_data(log_msg)
                        if not epoch_data:
                            continue

                        # Extract ZPE effects
                        zpe_effects = self._extract_zpe_effects(log_msg)
                        if zpe_effects is None:
                            zpe_effects = job_data.get('zpe_effects', [0.0] * 6)

                        # Log the epoch data
                        self.log_epoch(
                            job_id=job_id,
                            epoch=epoch_data['epoch'],
                            total_epochs=params.get('totalEpochs', 0),
                            timestamp=job_data.get('start_time', datetime.now().isoformat()),
                            model_name=params.get('modelName', ''),
                            base_config_id=params.get('baseConfigId', ''),
                            train_loss=epoch_data['train_loss'],
                            val_loss=epoch_data['val_loss'],
                            val_accuracy=epoch_data['val_accuracy'],
                            epoch_time=0.0,  # Not available in historical data
                            params=params,
                            zpe_effects=zpe_effects
                        )

                self._processed_jobs.add(job_id)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    def _extract_epoch_data(self, log_msg: str) -> Optional[Dict[str, Any]]:
        """Extract epoch data from a log message."""
        import re
        epoch_match = re.search(r'E(\d+)\s+END\s+-\s+TrainL:\s+([\d.]+),\s+ValAcc:\s+([\d.]+)%,\s+ValL:\s+([\d.]+)', log_msg)
        if not epoch_match:
            return None
        
        return {
            'epoch': int(epoch_match.group(1)),
            'train_loss': float(epoch_match.group(2)),
            'val_accuracy': float(epoch_match.group(3)),
            'val_loss': float(epoch_match.group(4))
        }

    def _extract_zpe_effects(self, log_msg: str) -> Optional[List[float]]:
        """Extract ZPE effects from a log message."""
        import re
        zpe_match = re.search(r'ZPE:\s+\[([\d.,\s]+)\]', log_msg)
        if not zpe_match:
            return None
        
        try:
            return [float(x.strip()) for x in zpe_match.group(1).split(',')]
        except:
            return None

    def log_epoch(self, job_id: str, epoch: int, total_epochs: int, timestamp: str,
                 model_name: str, base_config_id: str, train_loss: float, val_loss: float,
                 val_accuracy: float, epoch_time: float, params: Dict[str, Any],
                 zpe_effects: List[float]):
        """Log a single epoch of training data."""
        # Ensure all parameter arrays are of length 6
        momentum_params = params.get('momentumParams', [0.0] * 6)
        strength_params = params.get('strengthParams', [0.0] * 6)
        noise_params = params.get('noiseParams', [0.0] * 6)
        coupling_params = params.get('couplingParams', [0.0] * 6)

        for param_list in [momentum_params, strength_params, noise_params, coupling_params]:
            if len(param_list) < 6:
                param_list.extend([0.0] * (6 - len(param_list)))

        row = [
            job_id,
            epoch,
            total_epochs,
            timestamp,
            model_name,
            base_config_id,
            train_loss,
            val_loss,
            val_accuracy,
            epoch_time,
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

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_job_completion(self, job_id: str, final_accuracy: float, final_loss: float):
        """Log the completion of a training job."""
        # This method can be used to add any final job statistics
        pass 