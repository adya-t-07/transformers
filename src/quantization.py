import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

def load_nf4_compressed_model(model_id: str):
    """
    Loads a base model directly into hardware-accelerated 4-bit NormalFloat (NF4).
    Enables Double Quantization to squeeze the memory footprint further.
    """
    print(f"Loading base model [{model_id}] with true NF4 quantization...")
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Optimal 4-bit distribution for weights
        bnb_4bit_use_double_quant=True,       # Quantizes quantization constants to save extra VRAM
        bnb_4bit_compute_dtype=torch.float16  # Forward pass activations run in FP16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=nf4_config,
        device_map="auto"                     # Disperses layers across GPU registers efficiently
    )
    
    return model