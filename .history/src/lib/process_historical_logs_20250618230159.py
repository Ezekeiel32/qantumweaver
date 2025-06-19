import os import json
import csv
from datetime import datetime
import re
from typing import Dict, Any, List

def extract_epoch_data(log_msg: str) -> Dict[str, Any]:
    """Extract epoch data from a log message."""
    epoch_match = re.search(r'E(\d+)\s+END\s+-\s+TrainL:\s+([\d.]+),\s+ValAcc:\s+([\d.]+)%,\s+ValL:\s+([\d.]+)', log_msg)
    if not epoch_match:
        return None
    
    return {
        'epoch': int(epoch_match.group(1)),
        'train_loss': float(epoch_match.group(2)),
        'val_accuracy': float(epoch_match.group(3)),
        'val_loss': float(epoch_match.group(4))
    }

def extract_zpe_effects(log_msg: str) -> List[float]:
    """Extract ZPE effects from a log message."""
    zpe_match = re.search(r'ZPE:\s+\[([\d.,\s]+)\]', log_msg)
    if not zpe_match:
        return None
    
    try:
        return [float(x.strip()) for x in zpe_match.group(1).split(',')]
    except:
        return None

def process_historical_logs(logs_dir: str = "logs.json", csv_path: str = "training_logs.csv"):
    """Process all historical logs and add them to the CSV file."""
    # Ensure CSV exists with headers
    headers = [
        'Job_ID', 'Epoch', 'Total_Epochs', 'Timestamp', 'Model_Name', 'Base_Config_ID',
        'Train_Loss', 'Val_Loss', 'Val_Accuracy', 'Epoch_Time', 'Conv1_Channels',
        'Conv2_Channels', 'Conv3_Channels', 'Conv4_Channels', 'FC_Hidden', 'Dropout_FC',
        'Energy_Scale', 'Learning_Rate', 'Mixup_Alpha', 'Batch_Size', 'Momentum_Factor',
        'Label_Smoothing', 'Quantum_Circuit_Size', 'Quantum_Mode', 'Momentum_Params',
        'Strength_Params', 'Noise_Params', 'Coupling_Params', 'Cycle_Scale',
        'zpe_effects', 'Weight_Decay'
    ]
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    # Get existing job IDs from CSV
    existing_jobs = set()
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        existing_jobs = {row['Job_ID'] for row in reader}

    # Process each log file
    for filename in os.listdir(logs_dir):
        if not filename.endswith('.json'):
            continue

        job_id = filename.replace('.json', '')
        if job_id in existing_jobs:
            print(f"Skipping {job_id} - already in CSV")
            continue

        try:
            with open(os.path.join(logs_dir, filename), 'r') as f:
                job_data = json.load(f)

            if job_data.get('status') != 'completed':
                print(f"Skipping {job_id} - not completed")
                continue

            print(f"Processing {job_id}...")
            params = job_data.get('parameters', {})
            
            # Process each epoch
            for i in range(len(job_data.get('log_messages', []))):
                log_msg = job_data['log_messages'][i]
                epoch_data = extract_epoch_data(log_msg)
                
                if not epoch_data:
                    continue

                # Find corresponding ZPE effects
                zpe_effects = None
                for j in range(i, len(job_data['log_messages'])):
                    if f'E{epoch_data["epoch"]} END' in job_data['log_messages'][j]:
                        zpe_effects = extract_zpe_effects(job_data['log_messages'][j])
                        if zpe_effects:
                            break

                if zpe_effects is None:
                    zpe_effects = job_data.get('zpe_effects', [0.0] * 6)

                # Ensure all parameter arrays are of length 6
                momentum_params = params.get('momentumParams', [0.0] * 6)
                strength_params = params.get('strengthParams', [0.0] * 6)
                noise_params = params.get('noiseParams', [0.0] * 6)
                coupling_params = params.get('couplingParams', [0.0] * 6)

                for param_list in [momentum_params, strength_params, noise_params, coupling_params]:
                    if len(param_list) < 6:
                        param_list.extend([0.0] * (6 - len(param_list)))

                # Prepare row data
                row = [
                    job_id,
                    epoch_data['epoch'],
                    params.get('totalEpochs', 0),
                    job_data.get('start_time', datetime.now().isoformat()),
                    params.get('modelName', ''),
                    params.get('baseConfigId', ''),
                    epoch_data['train_loss'],
                    epoch_data['val_loss'],
                    epoch_data['val_accuracy'],
                    0.0,  # Epoch time not available in historical data
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
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            print(f"Completed processing {job_id}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    process_historical_logs() 