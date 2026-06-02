import torch
import gc
import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Safely loads the project YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def track_vram(label: str = "Current State"):
    """
    Logs active and peak GPU memory allocations. 
    Crucial for benchmarking the savings of Quantization and LoRA.
    """
    if torch.cuda.is_available():
        # Clear PyTorch's internal cache to get an accurate reading
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 3) # Convert bytes to GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        print(f"\n==== VRAM Telemetry: {label} ====")
        print(f"-> Active VRAM Allocation: {allocated:.3f} GB")
        print(f"-> Peak VRAM Allocated:    {max_allocated:.3f} GB")
        print("=================================\n")
    else:
        print(f"\n[Telemetry: {label}] No CUDA GPU detected. Running on CPU.")