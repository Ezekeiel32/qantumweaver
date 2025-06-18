import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# The exact 31 columns as provided by the user
COLUMNS = [
    'Job_ID','Epoch','Total_Epochs','Timestamp','Model_Name','Base_Config_ID','Train_Loss','Val_Loss','Val_Accuracy','Epoch_Time',
    'Conv1_Channels','Conv2_Channels','Conv3_Channels','Conv4_Channels','FC_Hidden','Dropout_FC','Energy_Scale','Learning_Rate','Mixup_Alpha','Batch_Size',
    'Momentum_Factor','Label_Smoothing','Quantum_Circuit_Size','Quantum_Mode','Momentum_Params','Strength_Params','Noise_Params','Coupling_Params',
    'Cycle_Scale','zpe_effects','Weight_Decay'
]

def extract_log_data(log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract and format job data for the 31-column CSV."""
    try:
        # Basic fields
        job_id = log_data.get('job_id', '')
        epoch = log_data.get('epoch', '')  # If not present, leave blank
        timestamp = log_data.get('timestamp', datetime.now().isoformat())
        model_name = log_data.get('model_name', '')
        base_config_id = log_data.get('base_config_id', '') or log_data.get('baseConfigId', '')
        
        # Metrics
        metrics = log_data.get('metrics', {})
        train_loss = metrics.get('train_loss', '')
        val_loss = metrics.get('val_loss', '')
        val_accuracy = metrics.get('val_accuracy', '')
        epoch_time = metrics.get('epoch_time', '')
        
        # Model/arch params
        params = log_data.get('parameters', {})
        total_epochs = params.get('totalEpochs', '')
        conv1_channels = params.get('conv1Channels', '')
        conv2_channels = params.get('conv2Channels', '')
        conv3_channels = params.get('conv3Channels', '')
        conv4_channels = params.get('conv4Channels', '')
        fc_hidden = params.get('fcHidden', '')
        dropout_fc = params.get('dropoutFC', '')
        energy_scale = params.get('energyScale', '')
        learning_rate = params.get('learningRate', '')
        mixup_alpha = params.get('mixupAlpha', '')
        batch_size = params.get('batchSize', '')
        momentum_factor = params.get('momentumFactor', '')
        label_smoothing = params.get('labelSmoothing', '')
        quantum_circuit_size = params.get('quantumCircuitSize', '')
        quantum_mode = params.get('quantumMode', '')
        cycle_scale = params.get('cycleScale', '')
        weight_decay = params.get('weightDecay', '')
        
        # Array params (store as stringified lists)
        def arr_to_str(arr):
            if isinstance(arr, list):
                return str(arr)
            return str(arr) if arr else ''
        momentum_params = arr_to_str(params.get('momentumParams', []))
        strength_params = arr_to_str(params.get('strengthParams', []))
        noise_params = arr_to_str(params.get('noiseParams', []))
        coupling_params = arr_to_str(params.get('couplingParams', []))
        zpe_effects = arr_to_str(log_data.get('zpe_effects', []))
        
        # Compose the row
        row = {
            'Job_ID': job_id,
            'Epoch': epoch,
            'Total_Epochs': total_epochs,
            'Timestamp': timestamp,
            'Model_Name': model_name,
            'Base_Config_ID': base_config_id,
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Val_Accuracy': val_accuracy,
            'Epoch_Time': epoch_time,
            'Conv1_Channels': conv1_channels,
            'Conv2_Channels': conv2_channels,
            'Conv3_Channels': conv3_channels,
            'Conv4_Channels': conv4_channels,
            'FC_Hidden': fc_hidden,
            'Dropout_FC': dropout_fc,
            'Energy_Scale': energy_scale,
            'Learning_Rate': learning_rate,
            'Mixup_Alpha': mixup_alpha,
            'Batch_Size': batch_size,
            'Momentum_Factor': momentum_factor,
            'Label_Smoothing': label_smoothing,
            'Quantum_Circuit_Size': quantum_circuit_size,
            'Quantum_Mode': quantum_mode,
            'Momentum_Params': momentum_params,
            'Strength_Params': strength_params,
            'Noise_Params': noise_params,
            'Coupling_Params': coupling_params,
            'Cycle_Scale': cycle_scale,
            'zpe_effects': zpe_effects,
            'Weight_Decay': weight_decay
        }
        return row
    except Exception as e:
        print(f"Error extracting log data: {e}")
        return None

def migrate_previous_runs():
    """Migrate all previous completed runs to the new 31-column CSV format."""
    log_dir = 'logs.json'
    master_log_path = 'zpe_master_log.csv'
    os.makedirs(log_dir, exist_ok=True)
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    entries = []
    for log_file in log_files:
        try:
            with open(os.path.join(log_dir, log_file), 'r') as f:
                log_data = json.load(f)
                if log_data.get('status') == 'completed':
                    entry = extract_log_data(log_data)
                    if entry:
                        entries.append(entry)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    if entries:
        df = pd.DataFrame(entries)
        df = df[COLUMNS]  # Ensure correct column order and only 31 columns
        df.to_csv(master_log_path, index=False)
        print(f"Successfully migrated {len(entries)} completed runs to {master_log_path} in 31-column format.")
        # Validate
        try:
            validation_df = pd.read_csv(master_log_path)
            if list(validation_df.columns) == COLUMNS:
                print("Validation passed: Column order and count match exactly.")
            else:
                print(f"Warning: Column mismatch. Got: {list(validation_df.columns)}")
        except Exception as e:
            print(f"Error validating output: {e}")
    else:
        print("No completed runs found to migrate.")

if __name__ == "__main__":
    print("Starting migration of previous runs to 31-column format...")
    migrate_previous_runs() 