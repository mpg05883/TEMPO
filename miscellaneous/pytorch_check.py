import torch

# Check PyTorch version
print("PyTorch version: ", torch.__version__)

# Check if CUDA is available
print("Is CUDA available? ", torch.cuda.is_available())

# Check CUDA version
print("CUDA version: ", torch.version.cuda)


# Check if CUDA is available
if torch.cuda.is_available():
    # Print the number of GPUs available
    print("Number of GPUs available: ", torch.cuda.device_count())
else:
    print("No GPUs available.")
