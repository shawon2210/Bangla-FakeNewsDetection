import torch
import os

print("Diagnosing model architectures...")

# List of model files to inspect
model_paths = [
    'outputs/best_multimodal_model.pth',
    'outputs/checkpoints/checkpoint_epoch_5.pth',
    'multimodal_detector_v1.pth'
]

# Loop through each model path and diagnose it
for i, path in enumerate(model_paths, 1):
    print(f"--- Model {i}: {path} ---")
    if not os.path.exists(path):
        print(f"Error: File not found at '{path}'")
        print("-" * (20 + len(path)))
        continue

    try:
        # Load the model file. The original command used weights_only=False, 
        # which can be a security risk if the file is not from a trusted source.
        model_data = torch.load(path, map_location='cpu', weights_only=False)

        print(f"Successfully loaded. Type: {type(model_data)}")

        keys_to_sample = []
        # Check if the loaded data is a dictionary (common for checkpoints)
        if isinstance(model_data, dict):
            if 'model_state_dict' in model_data:
                print("Found 'model_state_dict' key. Sampling keys from it.")
                keys_to_sample = list(model_data['model_state_dict'].keys())
            else:
                print("Loaded data is a dictionary. Sampling its keys.")
                keys_to_sample = list(model_data.keys())
        # Check if it's a model object with a state_dict method
        elif hasattr(model_data, 'state_dict'):
            print("Loaded data is a model object. Sampling keys from its state_dict.")
            keys_to_sample = list(model_data.state_dict().keys())
        else:
            print("Cannot determine keys for this model type.")

        if keys_to_sample:
            print(f"Sample of keys: {keys_to_sample[:5]}")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
    
    print("-" * (20 + len(path)))

print("Model diagnosis complete.")
