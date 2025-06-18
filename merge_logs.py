import json
import pandas as pd
import os
import glob

def read_json_log(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_log_data(log_data):
    try:
        job_id = log_data.get('job_id', '')
        timestamp = log_data.get('timestamp', '')
        model_name = log_data.get('model_name', '')
        status = log_data.get('status', '')
        metrics = log_data.get('metrics', {})
        final_accuracy = metrics.get('final_accuracy', 0.0)
        final_loss = metrics.get('final_loss', 0.0)
        zpe_effects = log_data.get('zpe_effects', [0.0]*6)
        if len(zpe_effects) < 6:
            zpe_effects = list(zpe_effects) + [0.0]*(6-len(zpe_effects))
        params = log_data.get('parameters', {})
        total_epochs = params.get('totalEpochs', 0)
        batch_size = params.get('batchSize', 0)
        learning_rate = params.get('learningRate', 0.0)
        weight_decay = params.get('weightDecay', 0.0)
        momentum_params = params.get('momentumParams', [0.0]*6)
        if len(momentum_params) < 6:
            momentum_params = list(momentum_params) + [0.0]*(6-len(momentum_params))
        strength_params = params.get('strengthParams', [0.0]*6)
        if len(strength_params) < 6:
            strength_params = list(strength_params) + [0.0]*(6-len(strength_params))
        noise_params = params.get('noiseParams', [0.0]*6)
        if len(noise_params) < 6:
            noise_params = list(noise_params) + [0.0]*(6-len(noise_params))
        coupling_params = params.get('couplingParams', [0.0]*6)
        if len(coupling_params) < 6:
            coupling_params = list(coupling_params) + [0.0]*(6-len(coupling_params))
        quantum_circuit_size = params.get('quantumCircuitSize', 0)
        label_smoothing = params.get('labelSmoothing', 0.0)
        quantum_mode = params.get('quantumMode', False)
        base_config_id = params.get('baseConfigId', '')
        return {
            'Job_ID': job_id,
            'Final_Acc': final_accuracy,
            'Final_Loss': final_loss,
            'ZPE_0': zpe_effects[0],
            'ZPE_1': zpe_effects[1],
            'ZPE_2': zpe_effects[2],
            'ZPE_3': zpe_effects[3],
            'ZPE_4': zpe_effects[4],
            'ZPE_5': zpe_effects[5],
            'Mom_0': momentum_params[0],
            'Mom_1': momentum_params[1],
            'Mom_2': momentum_params[2],
            'Mom_3': momentum_params[3],
            'Mom_4': momentum_params[4],
            'Mom_5': momentum_params[5],
            'Str_0': strength_params[0],
            'Str_1': strength_params[1],
            'Str_2': strength_params[2],
            'Str_3': strength_params[3],
            'Str_4': strength_params[4],
            'Str_5': strength_params[5],
            'Noise_0': noise_params[0],
            'Noise_1': noise_params[1],
            'Noise_2': noise_params[2],
            'Noise_3': noise_params[3],
            'Noise_4': noise_params[4],
            'Noise_5': noise_params[5],
            'Coup_0': coupling_params[0],
            'Coup_1': coupling_params[1],
            'Coup_2': coupling_params[2],
            'Coup_3': coupling_params[3],
            'Coup_4': coupling_params[4],
            'Coup_5': coupling_params[5],
            'Quantum_Circ_Size': quantum_circuit_size,
            'Label_Smoothing': label_smoothing,
            'Quantum_Mode': quantum_mode,
            'Model_Name': model_name,
            'Base_Config_ID': base_config_id,
            'Total_Epochs': total_epochs,
            'Batch_Size': batch_size,
            'Learning_Rate': learning_rate,
            'Weight_Decay': weight_decay,
            'Status': status,
            'Timestamp': timestamp
        }
    except Exception as e:
        print(f"Error processing log file: {e}")
        return None

def merge_logs():
    master_log_path = 'zpe_master_log.csv'
    if os.path.exists(master_log_path):
        master_df = pd.read_csv(master_log_path)
    else:
        master_df = pd.DataFrame()
    log_files = glob.glob('logs.json/zpe_job_*.json')
    new_entries = []
    for log_file in log_files:
        try:
            log_data = read_json_log(log_file)
            if log_data:
                entry = extract_log_data(log_data)
                if entry:
                    new_entries.append(entry)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        # Ensure column order matches the screenshot
        columns = [
            'Job_ID', 'Final_Acc', 'Final_Loss',
            'ZPE_0', 'ZPE_1', 'ZPE_2', 'ZPE_3', 'ZPE_4', 'ZPE_5',
            'Mom_0', 'Mom_1', 'Mom_2', 'Mom_3', 'Mom_4', 'Mom_5',
            'Str_0', 'Str_1', 'Str_2', 'Str_3', 'Str_4', 'Str_5',
            'Noise_0', 'Noise_1', 'Noise_2', 'Noise_3', 'Noise_4', 'Noise_5',
            'Coup_0', 'Coup_1', 'Coup_2', 'Coup_3', 'Coup_4', 'Coup_5',
            'Quantum_Circ_Size', 'Label_Smoothing', 'Quantum_Mode',
            'Model_Name', 'Base_Config_ID', 'Total_Epochs', 'Batch_Size',
            'Learning_Rate', 'Weight_Decay', 'Status', 'Timestamp'
        ]
        new_df = new_df[columns]
        if not master_df.empty:
            master_df = pd.concat([master_df, new_df], ignore_index=True)
        else:
            master_df = new_df
        master_df.to_csv(master_log_path, index=False)
        print(f"Successfully merged {len(new_entries)} new entries into master log")
    else:
        print("No new entries to merge")

def validate_master_log():
    try:
        df = pd.read_csv('zpe_master_log.csv')
        columns = [
            'Job_ID', 'Final_Acc', 'Final_Loss',
            'ZPE_0', 'ZPE_1', 'ZPE_2', 'ZPE_3', 'ZPE_4', 'ZPE_5',
            'Mom_0', 'Mom_1', 'Mom_2', 'Mom_3', 'Mom_4', 'Mom_5',
            'Str_0', 'Str_1', 'Str_2', 'Str_3', 'Str_4', 'Str_5',
            'Noise_0', 'Noise_1', 'Noise_2', 'Noise_3', 'Noise_4', 'Noise_5',
            'Coup_0', 'Coup_1', 'Coup_2', 'Coup_3', 'Coup_4', 'Coup_5',
            'Quantum_Circ_Size', 'Label_Smoothing', 'Quantum_Mode',
            'Model_Name', 'Base_Config_ID', 'Total_Epochs', 'Batch_Size',
            'Learning_Rate', 'Weight_Decay', 'Status', 'Timestamp'
        ]
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        print("Master log validation passed successfully")
        return True
    except Exception as e:
        print(f"Error validating master log: {e}")
        return False

if __name__ == "__main__":
    print("Starting log merge process...")
    merge_logs()
    print("\nValidating master log...")
    validate_master_log() 