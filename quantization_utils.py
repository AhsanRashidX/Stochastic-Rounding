# quantization_utils.py
import torch

def compute_scale(x, bits):
    abs_max = x.abs().max()
    
    if abs_max == 0 or torch.isinf(abs_max) or torch.isnan(abs_max):
        abs_max = x.abs().mean() * 3
    
    # For very low bits, be slightly conservative
    if bits <= 4:
        # Use 98th percentile to avoid extreme outlier sensitivity
        flat = x.abs().flatten()
        k = max(1, int(0.98 * flat.numel()))
        abs_max = torch.kthvalue(flat, k).values
    
    return abs_max / (2 ** (bits - 1) - 1)

def stochastic_quantize(x, bits, scale=None):
    if scale is None:
        scale = compute_scale(x, bits)
    
    # Ensure scale is never too small
    scale = max(scale, 1e-6)
    
    x_scaled = x / scale
    qmin = -(2 ** (bits - 1) - 1)
    qmax = 2 ** (bits - 1) - 1
    x_scaled = torch.clamp(x_scaled, qmin, qmax)
    
    floor = torch.floor(x_scaled)
    prob = x_scaled - floor
    rand = torch.rand_like(x_scaled, device=x.device)
    x_quantized = torch.where(rand < prob, floor + 1, floor)
    x_quantized = torch.clamp(x_quantized, qmin, qmax)
    
    return x_quantized * scale, scale

def deterministic_quantize(x, bits, scale=None):
    """Round-to-nearest quantization"""
    if scale is None:
        scale = compute_scale(x, bits)
    
    scale = max(scale, 1e-8)
    x_scaled = x / scale
    
    qmin = -(2 ** (bits - 1) - 1)
    qmax = 2 ** (bits - 1) - 1
    x_scaled = torch.clamp(x_scaled, qmin, qmax)
    
    x_quantized = torch.round(x_scaled)
    x_quantized = torch.clamp(x_quantized, qmin, qmax)
    
    return x_quantized * scale, scale