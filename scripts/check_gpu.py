import torch

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✅ GPU is ready to use!")
else:
    print("❌ GPU not detected")
