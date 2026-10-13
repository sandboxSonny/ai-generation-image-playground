import platform

import torch

print(f"PyTorch version: {torch.__version__}")

if platform.system() == "Darwin":
    print(f"MPS available: {torch.backends.mps.is_available()}")
    try:
        import mlx.core as mx
        print(f"MLX default device: {mx.default_device()}")
        print("✅ MLX is ready to use!")
    except ImportError:
        print("❌ MLX not installed (pip install -r requirements-mlx.txt)")
else:
    print(f"ROCm/CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("✅ GPU is ready to use!")
    else:
        print("❌ GPU not detected")
