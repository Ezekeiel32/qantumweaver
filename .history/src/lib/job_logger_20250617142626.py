import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

class JobLogger:
    def __init__(self, log_dir: str = 'logs.json', master_log_path: str = 'zpe_master_log.csv'):
        self.log_dir = log_dir
        self.master_log_path = master_log_path
        os.makedirs(log_dir, exist_ok=True)

    def _extract_log_data(self, job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and format job data for CSV logging."""
        try:
            job_id = job_data.get('job_id', '')
            timestamp = job_data.get('timestamp', datetime.now().isoformat())
            model_name = job_data.get('model_name', '')
            status = job_data.get('status', '')
            metrics = job_data.get('metrics', {})
            final_accuracy = metrics.get('final_accuracy', 0.0)
            final_loss = metrics.get('final_loss', 0.0)
            
            # Handle ZPE effects
            zpe_effects = job_data.get('zpe_effects', [0.0]*6)
            if len(zpe_effects) < 6:
                zpe_effects = list(zpe_effects) + [0.0]*(6-len(zpe_effects))
            
            # Handle parameters
            params = job_data.get('parameters', {})
            total_epochs = params.get('totalEpochs', 0)
            batch_size = params.get('batchSize', 0)
            learning_rate = params.get('learningRate', 0.0)
            weight_decay = params.get('weightDecay', 0.0)
            
            # Handle parameter arrays
            def pad_array(arr: List[float], length: int = 6) -> List[float]:
                if len(arr) < length:
                    return list(arr) + [0.0]*(length-len(arr))
                return arr[:length]
            
            momentum_params = pad_array(params.get('momentumParams', [0.0]*6))
            strength_params = pad_array(params.get('strengthParams', [0.0]*6))
            noise_params = pad_array(params.get('noiseParams', [0.0]*6))
            coupling_params = pad_array(params.get('couplingParams', [0.0]*6))
            
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
            print(f"Error extracting log data: {e}")
            return None

    def log_job(self, job_data: Dict[str, Any]) -> bool:
        """Log a single job to both JSON and CSV formats."""
        try:
            # Save JSON log
            job_id = job_data.get('job_id', '')
            if not job_id:
                print("Error: No job_id provided")
                return False
                
            json_path = os.path.join(self.log_dir, f'zpe_job_{job_id}.json')
            with open(json_path, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            # Extract data for CSV
            log_entry = self._extract_log_data(job_data)
            if not log_entry:
                return False
            
            # Update master CSV log
            if os.path.exists(self.master_log_path):
                master_df = pd.read_csv(self.master_log_path)
            else:
                master_df = pd.DataFrame()
            
            # Ensure column order
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
            
            new_df = pd.DataFrame([log_entry])
            new_df = new_df[columns]
            
            if not master_df.empty:
                master_df = pd.concat([master_df, new_df], ignore_index=True)
            else:
                master_df = new_df
            
            master_df.to_csv(self.master_log_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error logging job: {e}")
            return False

    def validate_master_log(self) -> bool:
        """Validate the master log CSV file."""
        try:
            if not os.path.exists(self.master_log_path):
                return True  # Empty log is valid
                
            df = pd.read_csv(self.master_log_path)
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
            return True
        except Exception as e:
            print(f"Error validating master log: {e}")
            return False 