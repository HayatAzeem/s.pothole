# Create a test.py file with this code to verify PyTorch GPU setup

import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No CUDA device available")
