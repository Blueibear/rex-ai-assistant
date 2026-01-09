"""Check if PyTorch can access the GPU for TTS."""

import sys

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Try to allocate a small tensor on GPU
        print("\nTesting GPU tensor allocation...")
        try:
            x = torch.randn(100, 100).cuda()
            print(f"✓ Successfully allocated tensor on GPU: {x.device}")
        except Exception as e:
            print(f"✗ Failed to allocate tensor on GPU: {e}")
    else:
        print("\n⚠️  GPU NOT AVAILABLE!")
        print("Possible reasons:")
        print("  1. CUDA not installed or not compatible with PyTorch")
        print("  2. GPU drivers not installed")
        print("  3. PyTorch CPU-only version installed")

        # Check if this is CPU-only PyTorch
        if not hasattr(torch.version, 'cuda') or torch.version.cuda is None:
            print("\n  → PyTorch was compiled without CUDA support")
            print("  → You have the CPU-only version of PyTorch")

except ImportError:
    print("ERROR: PyTorch is not installed")
    sys.exit(1)
