import torch
print("=== LIVE COLAB GPU STATUS ===")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name   :", torch.cuda.get_device_name(0))
    print("Device Count  :", torch.cuda.device_count())
    print("VRAM Allocated:", round(torch.cuda.memory_allocated(0)/(1024**2), 2), "MB")
    print("VRAM Reserved :", round(torch.cuda.memory_reserved(0)/(1024**2), 2), "MB")

