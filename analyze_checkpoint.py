#!/usr/bin/env python3
"""
Script to analyze the binary checkpoint file format and detect corruption.
"""

import struct
import os

def analyze_checkpoint(filepath):
    """Analyze the binary checkpoint file structure."""
    
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return
    
    file_size = os.path.getsize(filepath)
    print(f"File: {filepath}")
    print(f"File size: {file_size} bytes")
    print("-" * 50)
    
    try:
        with open(filepath, 'rb') as f:
            # Read metadata section
            print("=== METADATA SECTION ===")
            epoch = struct.unpack('i', f.read(4))[0]
            loss = struct.unpack('f', f.read(4))[0]
            accuracy = struct.unpack('f', f.read(4))[0]
            learning_rate = struct.unpack('f', f.read(4))[0]
            
            print(f"Epoch: {epoch}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}")
            print(f"Learning rate: {learning_rate}")
            
            # Read optimizer type
            print("\n=== OPTIMIZER SECTION ===")
            opt_len = struct.unpack('Q', f.read(8))[0]  # size_t is 8 bytes on 64-bit
            print(f"Optimizer string length: {opt_len}")
            
            if opt_len > 1000:  # Sanity check
                print(f"WARNING: Optimizer length seems too large: {opt_len}")
                return False
                
            optimizer_type = f.read(opt_len).decode('utf-8')
            print(f"Optimizer type: '{optimizer_type}'")
            
            # Read Adam parameters if using Adam
            if optimizer_type == "adam":
                print("\n=== ADAM PARAMETERS ===")
                beta1 = struct.unpack('f', f.read(4))[0]
                beta2 = struct.unpack('f', f.read(4))[0]
                epsilon = struct.unpack('f', f.read(4))[0]
                print(f"Beta1: {beta1}")
                print(f"Beta2: {beta2}")
                print(f"Epsilon: {epsilon}")
            
            # Read network structure
            print("\n=== NETWORK STRUCTURE ===")
            num_neurons = struct.unpack('Q', f.read(8))[0]
            num_inputs = struct.unpack('Q', f.read(8))[0]
            num_outputs = struct.unpack('Q', f.read(8))[0]
            time_steps = struct.unpack('Q', f.read(8))[0]
            
            print(f"Number of neurons: {num_neurons}")
            print(f"Number of inputs: {num_inputs}")
            print(f"Number of outputs: {num_outputs}")
            print(f"Time steps: {time_steps}")
            
            # Analyze first few neurons
            print("\n=== NEURON DATA ANALYSIS ===")
            current_pos = f.tell()
            remaining_bytes = file_size - current_pos
            print(f"Bytes used for metadata: {current_pos}")
            print(f"Remaining bytes for neuron data: {remaining_bytes}")
            
            # Try to read first neuron
            print("\n=== FIRST NEURON ANALYSIS ===")
            if remaining_bytes > 0:
                try:
                    # Read neuron ID
                    neuron_id = struct.unpack('i', f.read(4))[0]
                    print(f"First neuron ID: {neuron_id}")
                    
                    # Read neuron parameters
                    membrane_potential = struct.unpack('f', f.read(4))[0]
                    threshold = struct.unpack('f', f.read(4))[0]
                    leakage_ratio = struct.unpack('f', f.read(4))[0]
                    neuron_learning_rate = struct.unpack('f', f.read(4))[0]
                    time_step = struct.unpack('i', f.read(4))[0]  # int is 4 bytes, not 8
                    
                    print(f"Membrane potential: {membrane_potential}")
                    print(f"Threshold: {threshold}")
                    print(f"Leakage ratio: {leakage_ratio}")
                    print(f"Neuron learning rate: {neuron_learning_rate}")
                    print(f"Time step: {time_step}")
                    
                    # Read weight count - THIS IS WHERE THE CORRUPTION HAPPENS
                    pos_before_weights = f.tell()
                    num_weights = struct.unpack('Q', f.read(8))[0]
                    print(f"Number of weights: {num_weights}")
                    print(f"Position before weight count: {pos_before_weights}")
                    print(f"Position after weight count: {f.tell()}")
                    
                    if num_weights > 100000:
                        print(f"❌ CORRUPTION DETECTED: Weight count {num_weights} is unreasonably large!")
                        
                        # Try to find patterns in the data around this position
                        f.seek(pos_before_weights - 16)
                        context = f.read(32)
                        print(f"Data context (16 bytes before to 16 bytes after):")
                        print(" ".join([f"{b:02x}" for b in context]))
                        
                        # Try interpreting as different data types
                        f.seek(pos_before_weights)
                        as_int32 = struct.unpack('I', f.read(4))[0]
                        f.seek(pos_before_weights)
                        as_float = struct.unpack('f', f.read(4))[0]
                        f.seek(pos_before_weights)
                        as_int64 = struct.unpack('Q', f.read(8))[0]
                        
                        print(f"Same bytes interpreted as:")
                        print(f"  uint32: {as_int32}")
                        print(f"  float: {as_float}")
                        print(f"  uint64: {as_int64}")
                        
                        return False
                    else:
                        print(f"✅ Weight count looks reasonable: {num_weights}")
                        
                        # Calculate expected size for this neuron
                        weights_size = num_weights * 4  # 4 bytes per float
                        adam_states_size = num_weights * 4 * 2  # m_weights + v_weights
                        adam_threshold_size = 4 * 2  # m_threshold + v_threshold
                        total_neuron_size = 4 + 4*4 + 8 + 8 + weights_size + adam_states_size + adam_threshold_size
                        
                        print(f"Expected size per neuron: {total_neuron_size} bytes")
                        print(f"Expected total neuron data size: {total_neuron_size * num_neurons} bytes")
                        print(f"Actual remaining bytes: {remaining_bytes}")
                        
                        return True
                        
                except struct.error as e:
                    print(f"❌ Error reading neuron data: {e}")
                    return False
            else:
                print("❌ No remaining bytes for neuron data")
                return False
                
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return False

def main():
    checkpoint_path = r"D:\Workspace\NeuromorphicNet\NeuromorphicNet\checkpoints\MNIST\best_model.bin"
    print("Analyzing best_model.bin...")
    result1 = analyze_checkpoint(checkpoint_path)
    
    print("\n" + "="*80 + "\n")
    
    checkpoint_path2 = r"D:\Workspace\NeuromorphicNet\NeuromorphicNet\checkpoints\MNIST\checkpoint_epoch_0.bin"
    print("Analyzing checkpoint_epoch_0.bin...")
    result2 = analyze_checkpoint(checkpoint_path2)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"best_model.bin: {'✅ Valid' if result1 else '❌ Corrupted'}")
    print(f"checkpoint_epoch_0.bin: {'✅ Valid' if result2 else '❌ Corrupted'}")

if __name__ == "__main__":
    main()