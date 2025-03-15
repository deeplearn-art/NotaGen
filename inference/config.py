import os

# Configurations for inference
INFERENCE_WEIGHTS_PATH = ''               # Path to weights for inference
NUM_SAMPLES = 1000                        # Number of samples to generate (only for generate mode)

# Hyperparameters that can be changed from notebook
TOP_K = 9                                 # Top k for sampling
TOP_P = 0.9                               # Top p for sampling
TEMPERATURE = 1.2                         # Temperature for sampling

# Function to get current output folders based on current parameters
def get_output_folders():
    """Return output folder paths based on current parameter values"""
    weight_name = os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] if INFERENCE_WEIGHTS_PATH else 'unnamed'
    param_suffix = f'_k_{TOP_K}_p_{TOP_P}_temp_{TEMPERATURE}'
    
    original = os.path.join('../output/original', weight_name + param_suffix)
    interleaved = os.path.join('../output/interleaved', weight_name + param_suffix)
    
    return original, interleaved

# Define fixed variables for backward compatibility
ORIGINAL_OUTPUT_FOLDER, INTERLEAVED_OUTPUT_FOLDER = get_output_folders()

# Configurations for model
PATCH_STREAM = True                       # Stream training / inference
PATCH_SIZE = 16                           # Patch Size
PATCH_LENGTH = 1024                       # Patch Length
CHAR_NUM_LAYERS = 6                       # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                     # Number of layers in the encoder
HIDDEN_SIZE = 1280                        # Hidden Size