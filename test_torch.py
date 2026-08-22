import torch

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check for CUDA (GPU) availability
print(f"Is CUDA available?: {torch.cuda.is_available()}")