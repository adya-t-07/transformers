import torch
import torch.nn as nn
import math
import bitsandbytes as bnb

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha/r

        #freezing original weights
        self.original_layer.requires_grad_= False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        #creating low rank matrices
        in_feats = original_layer.in_features
        out_feats = original_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_feats, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_feats))

        #using kaiming uniform distribution to initialize A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) #look into why
    
    # def forward(self, x:torch.tensor)->torch.tensor:
    #     #forward pass through original layer
    #     base_output = self.original_layer(x)
    #     #forward pass through lora modified layer
    #     lora_output = (x @ self.lora_A) @ self.lora_B
    #     return base_output + (self.scaling * lora_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If original_layer is a Linear4bit module, it auto-computes the NF4 math in FP16
        base_output = self.original_layer(x)
        
        # Ensure our low-rank path matches the tensor's precision dynamically
        x_typed = x.to(self.lora_A.dtype)
        lora_output = (x_typed @ self.lora_A) @ self.lora_B
        
        return base_output + (lora_output * self.scaling)
    


# def inject_lora(model:nn.Module, target_modules=["q_proj", "v_proj"], r:int=8, alpha:int=16):
#     for param in model.parameters():
#         param.requires_grad = False
#     #iterate through children
#     for name, module in model.named_children():
#         #if the child is a linear layer and is a Q/V projection layer, we wrap it with lora
#         if name in target_modules and isinstance(module, nn.Linear):
#             wrapped_layer = LoRALinear(module, r, alpha)
#             setattr(model, name, wrapped_layer)

#         #if it's not a leaf module, recursively go through its children
#         elif len(list(module.named_children())) > 0:
#             inject_lora(module, target_modules, r, alpha)

def inject_lora(model: nn.Module, target_modules=["q_proj", "v_proj"], r: int = 8, alpha: int = 16):
    """
    Recursively scans the model, replacing standard Linear OR NF4 4-bit Linear layers
    with your custom LoRALinear wrappers.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            inject_lora(module, target_modules, r, alpha)
            
        for target in target_modules:
            if name == target:
                # Check for either standard PyTorch Linear layers OR hardware 4bit layers
                is_standard_linear = isinstance(module, nn.Linear)
                is_4bit_linear = isinstance(module, bnb.nn.Linear4bit)
                
                if is_standard_linear or is_4bit_linear:
                    # Inject your custom wrapper directly over the NF4 layer
                    wrapped_layer = LoRALinear(module, r=r, lora_alpha=alpha)
                    setattr(model, name, wrapped_layer)
