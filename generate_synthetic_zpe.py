import csv
import random
import numpy as np

def generate_zpe_flow_params(cycle_length=32):
    # Optimized ranges based on high-accuracy patterns
    momentum = random.uniform(0.75, 0.95)  # Higher momentum for stability
    strength = random.uniform(0.15, 0.35)  # Moderate strength for controlled effects
    noise = random.uniform(0.1, 0.25)      # Lower noise for stability
    coupling = random.uniform(0.65, 0.85)  # Balanced coupling
    
    # Quantum cycle modulation
    cycle_factor = (11/16) * (cycle_length / 32)
    false_ten = 3.33 * 3
    strength *= cycle_factor * (false_ten / 10)
    
    # Ensure parameters are within optimal ranges
    momentum = max(0.75, min(0.95, momentum))
    strength = max(0.15, min(0.35, strength))
    noise = max(0.1, min(0.25, noise))
    coupling = max(0.65, min(0.85, coupling))
    
    return momentum, strength, noise, coupling

def generate_extra_zpe():
    # Optimized ranges for additional ZPE effects
    zpe5 = random.uniform(0.15, 0.35)  # More controlled range
    zpe6 = random.uniform(0.15, 0.35)  # More controlled range
    return zpe5, zpe6

def generate_quantum_params():
    # Generate quantum-specific parameters
    quantum_circuit_size = random.choice([32, 48, 64])  # Larger circuits for better performance
    quantum_mode = random.random() < 0.7  # 70% chance of quantum mode
    return quantum_circuit_size, quantum_mode

csv_path = 'zpe_supervised_dataset.csv'

# Write header
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['job_id', 'epoch', 'momentum', 'strength', 'noise', 'coupling', 
                    'zpe5', 'zpe6', 'quantum_circuit_size', 'quantum_mode', 'accuracy'])

# Generate synthetic data
with open(csv_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(1_000_000):  # Reduced to 1M jobs for better quality
        job_id = f'synth_job_{i}'
        epoch = random.randint(8, 15)  # Match training epochs
        
        # Generate ZPE parameters
        momentum, strength, noise, coupling = generate_zpe_flow_params()
        zpe5, zpe6 = generate_extra_zpe()
        quantum_circuit_size, quantum_mode = generate_quantum_params()
        
        # Generate synthetic accuracy based on parameter quality
        base_accuracy = 0.85  # Base accuracy for good parameters
        
        # Adjust accuracy based on parameter quality
        momentum_factor = (momentum - 0.75) / 0.2  # Normalize to 0-1
        strength_factor = 1 - abs(strength - 0.25) / 0.2  # Peak at 0.25
        noise_factor = 1 - (noise - 0.1) / 0.15  # Lower noise is better
        coupling_factor = (coupling - 0.65) / 0.2  # Normalize to 0-1
        
        # Quantum factors
        quantum_factor = 1.1 if quantum_mode else 1.0
        circuit_factor = quantum_circuit_size / 64  # Normalize to 0-1
        
        # Calculate final accuracy with some randomness
        accuracy = base_accuracy * (
            0.3 * momentum_factor +
            0.2 * strength_factor +
            0.2 * noise_factor +
            0.2 * coupling_factor +
            0.1 * circuit_factor
        ) * quantum_factor
        
        # Add small random variation
        accuracy += random.uniform(-0.02, 0.02)
        accuracy = max(0.5, min(0.98, accuracy))  # Clamp between 50% and 98%
        
        writer.writerow([job_id, epoch, momentum, strength, noise, coupling, 
                        zpe5, zpe6, quantum_circuit_size, quantum_mode, accuracy])
        
        if i % 100000 == 0:
            print(f'Generated {i} jobs...')

print("Done! 1,000,000 synthetic ZPE jobs generated with optimized parameters.") 