# quantized_layers.py
import torch
import torch.nn as nn
from quantization_utils import stochastic_quantize, deterministic_quantize

# Import Conv1D from transformers
try:
    from transformers.modeling_utils import Conv1D
except ImportError:
    from transformers.pytorch_utils import Conv1D


class QuantizedLinearFunction(torch.autograd.Function):
    """
    Custom autograd function for mixed-precision training
    Forward: Quantized weights (QAT), FP32 activations
    Backward: Quantized activations & gradients (SR), FP32 weights
    """
    @staticmethod
    def forward(ctx, x, weight, bias, bits, use_sr):
        ctx.bits = bits
        ctx.use_sr = use_sr
        ctx.save_for_backward(x, weight, bias)
        
        if bits < 32:
            quant_fn = stochastic_quantize if use_sr else deterministic_quantize
            w_quant, _ = quant_fn(weight, bits)
            weight_use = weight + (w_quant - weight).detach()
        else:
            weight_use = weight
        
        output = torch.matmul(x, weight_use.t())
        if bias is not None:
            output = output + bias
            
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        bits = ctx.bits
        use_sr = ctx.use_sr
        
        quant_fn = stochastic_quantize if use_sr else deterministic_quantize
        
        x_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            x_2d = x
            grad_output_2d = grad_output
        
        x_quant, _ = quant_fn(x_2d, bits)
        grad_output_quant, _ = quant_fn(grad_output_2d, bits)
        
        grad_weight = torch.matmul(grad_output_quant.t(), x_quant)
        grad_input_2d = torch.matmul(grad_output_quant, weight)
        
        if x.dim() > 2:
            grad_input = grad_input_2d.reshape(x_shape)
        else:
            grad_input = grad_input_2d
        
        if bias is not None:
            grad_bias = grad_output.sum(dim=list(range(grad_output.dim()-1)))
        else:
            grad_bias = None
        
        return grad_input, grad_weight, grad_bias, None, None


class QuantizedLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bits=8, use_sr=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.use_sr = use_sr
        
        # Standard nn.Linear weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
    def forward(self, x):
        return QuantizedLinearFunction.apply(x, self.weight, self.bias, self.bits, self.use_sr)


def replace_linear_with_quantized(model, bits=8, use_sr=True, skip_lm_head=True, skip_embeddings=True):
    """
    Recursively replace ALL Linear AND Conv1D layers with quantized versions.
    Conv1D has transposed weight shape compared to nn.Linear!
    """
    modules_to_replace = []
    
    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = isinstance(module, Conv1D)
        
        if is_linear or is_conv1d:
            # Skip LM head if requested
            if skip_lm_head and 'lm_head' in name:
                print(f"  Skipping {name} (LM head)")
                continue
            
            # Skip embeddings if requested
            if skip_embeddings and ('wte' in name or 'wpe' in name or 'embed' in name):
                print(f"  Skipping {name} (embedding)")
                continue
            
            modules_to_replace.append((name, module, is_conv1d))
    
    # print(f"  Found {len(modules_to_replace)} layers to quantize ({sum(1 for _, _, c in modules_to_replace if c)} Conv1D, {sum(1 for _, _, c in modules_to_replace if not c)} Linear)")
    
    for name, module, is_conv1d in modules_to_replace:
        # Navigate to parent module
        *parent_path, attr_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        
        if is_conv1d:
            # Conv1D: weight shape is (in_features, out_features)
            # QuantizedLinearLayer: weight shape is (out_features, in_features)
            in_features = module.weight.shape[0]
            out_features = module.nf  # Conv1D stores output dim as self.nf
            
            q_layer = QuantizedLinearLayer(in_features, out_features, bits=bits, use_sr=use_sr)
            
            # TRANSPOSE the weight from Conv1D to nn.Linear format
            q_layer.weight.data = module.weight.data.t().clone()
            q_layer.bias.data = module.bias.data.clone()
            
            # print(f"  Quantized {name} (Conv1D): {in_features} -> {out_features}")
        else:
            # Standard nn.Linear
            q_layer = QuantizedLinearLayer(
                module.in_features, 
                module.out_features,
                bits=bits,
                use_sr=use_sr
            )
            q_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                q_layer.bias.data = module.bias.data.clone()
            else:
                q_layer.bias.data.zero_()
            
            # print(f"  Quantized {name} (Linear): {module.in_features} -> {module.out_features}")
        
        setattr(parent, attr_name, q_layer)
    
    return model