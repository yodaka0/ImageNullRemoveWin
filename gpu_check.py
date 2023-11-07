import torch

print(torch.backends.cudnn.version())
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")
