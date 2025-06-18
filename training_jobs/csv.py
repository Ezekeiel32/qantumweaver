import json
import csv
from datetime import datetime

# Load the JSON data from the file
with open("zpe_job_169cda4c.json") as file:
    content = file.read()
    
# Parse the JSON data
data = json.loads(content)

# Define the header for the CSV file
header = [
    'Job_ID', 'Epoch', 'Total_Epochs', 'Timestamp', 'Model_Name',
    'Base_Config_ID', 'Train_Loss', 'Val_Loss', 'Val_Accuracy',
    'Epoch_Time', 'Conv1_Channels', 'Conv2_Channels', 'Conv3_Channels',
    'Conv4_Channels', 'FC_Hidden', 'Dropout_FC', 'Energy_Scale',
    'Learning_Rate', 'Mixup_Alpha', 'Batch_Size', 'Momentum_Factor',
    'Label_Smoothing', 'Quantum_Circuit_Size', 'Quantum_Mode',
    'Momentum_Params', 'Strength_Params', 'Noise_Params',
    'Coupling_Params', 'Cycle_Scale', 'zpe_effects', 'Weight_Decay'
]

# Create the output CSV file
output_file = "training_results.csv"

# Write the data to the CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    
    # Extract parameters from the data
    job_id = data["job_id"]
    total_epochs = data["total_epochs"]
    model_name = data["model_name"]
    base_config_id = data["base_config_id"]
    conv_channels = data["conv_channels"]
    fc_hidden = data["fc_hidden"]
    dropout_fc = data["dropout_fc"]
    energy_scale = data["energy_scale"]
    learning_rate = data["learning_rate"]
    mixup_alpha = data["mixup_alpha"]
    batch_size = data["batch_size"]
    momentum_factor = data["momentum_factor"]
    label_smoothing = data["label_smoothing"]
    quantum_circuit_size = data["quantum_circuit_size"]
    quantum_mode = data["quantum_mode"]
    momentum_params = json.dumps(data["momentum_params"])
    strength_params = json.dumps(data["strength_params"])
    noise_params = json.dumps(data["noise_params"])
    coupling_params = json.dumps(data["coupling_params"])
    cycle_scale = data["cycle_scale"]
    weight_decay = data["weight_decay"]
    
    # Process each epoch
    for epoch_data in data["epochs"]:
        row = [
            job_id,
            epoch_data["epoch"],
            total_epochs,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            base_config_id,
            epoch_data["train_loss"],
            epoch_data["val_loss"],
            epoch_data["val_accuracy"],
            epoch_data["epoch_time"],
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
            quantum_mode,
            momentum_params,
            strength_params,
            noise_params,
            coupling_params,
            cycle_scale,
            json.dumps(epoch_data["zpe_effects"]),
            weight_decay
        ]
        
        writer.writerow(row)

print(f"CSV file '{output_file}' has been created successfully.")
