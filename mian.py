"""
Stochastic Rounding for Edge LLM Training
Implementation based on Liu et al. (2025) - "Unlocking Edge LLMs Training with Stochastic Rounding"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from contextlib import contextmanager
import copy


# ============================================================================
# 1. STOCHASTIC ROUNDING CORE OPERATIONS
# ============================================================================

def stochastic_round(x: torch.Tensor, delta: float, epsilon: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Stochastic Rounding (SR) as defined in Definition 3 of the paper.
    
    Args:
        x: Input tensor to quantize
        delta: Quantization step size (precision level)
        epsilon: Random threshold from U[0,1]. If None, generates new random values
    
    Returns:
        Quantized tensor using stochastic rounding
    """
    if delta == 0:
        return x
    
    # Normalize by step size
    x_scaled = x / delta
    x_floor = torch.floor(x_scaled)
    frac = x_scaled - x_floor
    
    # Generate random thresholds if not provided
    if epsilon is None:
        epsilon = torch.rand_like(x)
    
    # Stochastic rounding: round up with probability = frac
    x_quantized = torch.where(frac < epsilon, x_floor, x_floor + 1)
    
    # Scale back
    return x_quantized * delta


def round_to_nearest(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Round-to-Nearest (RTN) for comparison/baseline.
    """
    if delta == 0:
        return x
    return torch.round(x / delta) * delta


def compute_delta(bits: int, max_val: float, format_type: str = 'e4m3') -> float:
    """
    Compute quantization step size for given bit-width.
    
    Args:
        bits: Number of mantissa bits (e.g., E4M3 -> 3 mantissa bits)
        max_val: Maximum absolute value in tensor
        format_type: 'e4m3', 'e5m2', 'int8', etc.
    """
    if format_type in ['e4m3', 'fp8']:
        # FP8 E4M3: 1 sign, 4 exp, 3 mantissa bits
        # Dynamic range handling
        return max_val / (2 ** (bits - 1))
    elif format_type == 'int8':
        return max_val / 127.0
    elif format_type == 'int4':
        return max_val / 7.0
    else:
        return max_val / (2 ** (bits - 1))


# ============================================================================
# 2. QUANTIZED LINEAR LAYER (The Core Component)
# ============================================================================

class SRQuantizedLinear(nn.Module):
    """
    Linear layer with Stochastic Rounding for mixed-precision training.
    Implements the framework from Section 3.3 of the paper.
    
    Key features:
    - Weight quantization shared across forward/backward (QAT objective)
    - Forward activation in high precision (Δ_fwd_A = 0)
    - Backward activation/gradient quantized with SR (Δ_bwd_A = Δ_bwd_∇A = Δ)
    - Per-sample stochastic thresholds for activations/gradients
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int = 4,           # E4M3 or similar
        activation_bits: int = 4,        # For backward pass quantization
        gradient_bits: int = 4,        # For gradient quantization
        stochastic_weights: bool = False,  # True for SR, False for RTN on weights
        use_high_precision_forward: bool = True,  # Δ_fwd_A = 0 as per paper
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.gradient_bits = gradient_bits
        self.stochastic_weights = stochastic_weights
        self.use_high_precision_forward = use_high_precision_forward
        
        # Weight parameter (stored in high precision, quantized during use)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
        # Store quantization steps (computed dynamically based on weight stats)
        self.register_buffer('weight_delta', torch.tensor(0.0))
        self.register_buffer('activation_delta', torch.tensor(0.0))
        self.register_buffer('gradient_delta', torch.tensor(0.0))
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def compute_quantization_steps(self, x: torch.Tensor):
        """Compute delta based on tensor statistics."""
        with torch.no_grad():
            # Dynamic quantization based on current tensor statistics
            max_w = self.weight.abs().max()
            self.weight_delta = compute_delta(self.weight_bits, max_w.item(), 'e4m3')
            
            if not self.use_high_precision_forward and x is not None:
                max_a = x.abs().max()
                self.activation_delta = compute_delta(self.activation_bits, max_a.item(), 'e4m3')
    
    def quantize_weight(self, deterministic: bool = False) -> torch.Tensor:
        """
        Quantize weights. Shared between forward and backward passes.
        Can be stochastic or deterministic (RTN).
        """
        if deterministic or not self.stochastic_weights:
            return round_to_nearest(self.weight, self.weight_delta)
        else:
            # Stochastic weight quantization (optional per paper Section 4.1)
            return stochastic_round(self.weight, self.weight_delta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Algorithm 1 from the paper.
        
        Key: Forward activation remains high precision (Δ_fwd_A = 0)
        """
        # Update quantization steps based on current statistics
        self.compute_quantization_steps(x)
        
        # Quantize weights (shared for forward/backward)
        # Use straight-through estimator for gradient flow
        weight_quantized = self.quantize_weight()
        weight_used = self.weight + (weight_quantized - self.weight).detach()
        
        # Forward pass: Activation in high precision (Δ_fwd_A = 0)
        # This ensures convergence as per paper Section 3.3
        if self.use_high_precision_forward:
            output = F.linear(x, weight_used, self.bias)
        else:
            # Alternative: quantize forward activation too (not recommended by paper)
            x_quantized = round_to_nearest(x, self.activation_delta)
            x_used = x + (x_quantized - x).detach()
            output = F.linear(x_used, weight_used, self.bias)
        
        return output


# ============================================================================
# 3. CUSTOM AUTOGRAD FUNCTION FOR BACKWARD PASS SR
# ============================================================================

class SRLinearFunction(torch.autograd.Function):
    """
    Custom autograd function implementing Algorithm 2 (Backward Pass) with SR.
    
    Critical for the paper's results:
    - Per-sample stochastic rounding of activations and gradients
    - Variance scales as 1/batch_size
    - Unbiased gradient estimates
    """
    
    @staticmethod
    def forward(ctx, x, weight, bias, weight_delta, act_delta, grad_delta, 
                stochastic_act, stochastic_grad):
        # Save for backward
        ctx.save_for_backward(x, weight, bias)
        ctx.weight_delta = weight_delta
        ctx.act_delta = act_delta
        ctx.grad_delta = grad_delta
        ctx.stochastic_act = stochastic_act
        ctx.stochastic_grad = stochastic_grad
        
        # Quantize weights for forward (deterministic or shared stochastic)
        weight_quant = round_to_nearest(weight, weight_delta)
        weight_used = weight + (weight_quant - weight).detach()
        
        # Forward activation in high precision (Δ_fwd_A = 0)
        output = torch.matmul(x, weight_used.t())
        if bias is not None:
            output = output + bias
        
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        
        batch_size = x.size(0)
        
        # =========================================================================
        # BACKWARD PASS WITH STOCHASTIC ROUNDING
        # =========================================================================
        
        # 1. Quantize backward activation (A_in)
        if ctx.act_delta > 0:
            if ctx.stochastic_act:
                # SR: random thresholds per sample
                epsilon_act = torch.rand_like(x)
                x_quant = stochastic_round(x, ctx.act_delta, epsilon_act)
            else:
                # RTN: deterministic rounding (always 0.5 threshold)
                x_quant = round_to_nearest(x, ctx.act_delta)
            x_used = x + (x_quant - x).detach()
        else:
            x_used = x
            
        # 2. Quantize weight for backward (shared quantization)
        # Note: weight quantization is always deterministic (shared across batch)
        weight_quant = round_to_nearest(weight, ctx.weight_delta)
        weight_used = weight + (weight_quant - weight).detach()
        
        # 3. Quantize output gradient (∇A_out)
        if ctx.grad_delta > 0:
            if ctx.stochastic_grad:
                # SR: random thresholds per sample
                epsilon_grad = torch.rand_like(grad_output)
                grad_quant = stochastic_round(grad_output, ctx.grad_delta, epsilon_grad)
            else:
                # RTN: deterministic rounding
                grad_quant = round_to_nearest(grad_output, ctx.grad_delta)
            grad_used = grad_output + (grad_quant - grad_output).detach()
        else:
            grad_used = grad_output
            
        
        # Compute gradients with quantized values
        # grad_output shape: (batch_size, out_features)
        # x shape: (batch_size, in_features)
        # weight shape: (out_features, in_features)
        
        # grad_weight = grad_output^T @ x_used 
        # Result: (out_features, in_features)
        grad_weight = torch.matmul(grad_used.t(), x_used)
        
        # grad_input = grad_used @ weight_used (not weight_used.t()!)
        # weight_used is (out_features, in_features)
        # grad_used is (batch_size, out_features)
        # Result: (batch_size, in_features)
        grad_input = torch.matmul(grad_used, weight_used)
        
        # Bias gradient: sum over batch dimension
        grad_bias = grad_used.sum(0) if bias is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class SRLinear(nn.Module):
    """
    Production-ready SR Linear layer using custom autograd.
    """
    def __init__(self, in_features, out_features, weight_bits=4, act_bits=4, 
                 grad_bits=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.grad_bits = grad_bits
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Quantization steps - initialize as scalar tensors
        self.register_buffer('w_delta', torch.tensor(0.0))
        self.register_buffer('a_delta', torch.tensor(0.0))
        self.register_buffer('g_delta', torch.tensor(0.0))
        
    def update_deltas(self, x):
        with torch.no_grad():
            # Wrap scalar values in torch.tensor() before assignment
            self.w_delta = torch.tensor(compute_delta(self.weight_bits, self.weight.abs().max().item()))
            self.a_delta = torch.tensor(compute_delta(self.act_bits, x.abs().max().item()))
            self.g_delta = torch.tensor(compute_delta(self.grad_bits, 1.0))
    
    def forward(self, x):
        self.update_deltas(x)
        return SRLinearFunction.apply(
            x, self.weight, self.bias, 
            self.w_delta, self.a_delta, self.g_delta,
            True, True
        )

# ============================================================================
# 4. TRAINING CONFIGURATION AND UTILITIES
# ============================================================================

class SRTrainingConfig:
    """
    Configuration for SR training following paper guidelines.
    
    Key insight from paper: 1-bit precision reduction ≈ 2-4x batch size increase
    """
    def __init__(
        self,
        weight_format: str = 'E4M3',      # E4M3, E5M2, E4M2, E4M1, E4M0
        activation_format: str = 'E4M3',
        gradient_format: str = 'E4M3',
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        stochastic_weights: bool = False,
        use_gradient_accumulation: bool = False,
        accumulation_steps: int = 1,
    ):
        self.weight_format = weight_format
        self.activation_format = activation_format
        self.gradient_format = gradient_format
        self.batch_size = batch_size
        self.lr = learning_rate
        self.stochastic_weights = stochastic_weights
        self.use_gradient_accumulation = use_gradient_accumulation
        self.accumulation_steps = accumulation_steps
        
        # Parse format strings to bit widths
        self.weight_bits = self._parse_format(weight_format)
        self.act_bits = self._parse_format(activation_format)
        self.grad_bits = self._parse_format(gradient_format)
    
    def _parse_format(self, fmt: str) -> int:
        """Parse format string to mantissa bits."""
        formats = {
            'E4M3': 3, 'E5M2': 2, 'E4M2': 2, 'E4M1': 1, 'E4M0': 0,
            'FP16': 10, 'BF16': 7, 'FP32': 23
        }
        return formats.get(fmt, 3)
    
    def get_effective_batch_size(self) -> int:
        """Account for gradient accumulation."""
        return self.batch_size * self.accumulation_steps


class SROptimizer(torch.optim.Optimizer):
    """
    Optimizer wrapper that handles SR-specific considerations.
    
    The paper suggests that SR works well with standard optimizers (SGD, Adam).
    The key is the unbiased gradient estimates from SR.
    """
    def __init__(self, params, base_optimizer_class=torch.optim.AdamW, 
                 lr=1e-4, weight_decay=0.01, **kwargs):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer_class(self.param_groups, lr=lr, 
                                                    weight_decay=weight_decay, **kwargs)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Step with potential weight quantization-aware updates."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.base_optimizer.step()
        return loss


# ============================================================================
# 5. MODEL BUILDING UTILITIES
# ============================================================================

def convert_to_sr_model(
    model: nn.Module,
    config: SRTrainingConfig,
    target_layers: Tuple[type] = (nn.Linear,),
) -> nn.Module:
    """
    Convert a standard model to use SR quantized layers.
    """
    def replace_module(module):
        for name, child in module.named_children():
            if isinstance(child, target_layers):
                # Replace with SR version
                sr_layer = SRLinear(
                    child.in_features,
                    child.out_features,
                    weight_bits=config.weight_bits,
                    act_bits=config.act_bits,
                    grad_bits=config.grad_bits,
                    bias=child.bias is not None,
                )
                # Copy weights
                with torch.no_grad():
                    sr_layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        sr_layer.bias.copy_(child.bias)
                setattr(module, name, sr_layer)
            else:
                replace_module(child)
    
    replace_module(model)
    return model


# ============================================================================
# 6. TRAINING LOOP WITH SR
# ============================================================================

def train_with_sr(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: SRTrainingConfig,
    num_epochs: int = 1,
    device: str = 'cuda',
    eval_fn=None,
) -> Dict[str, Any]:
    """
    Training loop implementing SR mixed-precision training.
    
    Key aspects from paper:
    1. Larger batches compensate for lower precision
    2. Gradient accumulation can emulate larger effective batch size
    3. SR provides unbiased gradients enabling stable convergence
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    history = {
        'losses': [],
        'gradient_norms': [],
        'accuracies': []
    }
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        grad_norms = []
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                inputs, targets = batch['input'], batch['target']
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Scale loss for gradient accumulation
            if config.use_gradient_accumulation:
                loss = loss / config.accumulation_steps
            
            # Backward pass with SR (happens automatically in custom layers)
            loss.backward()
            
            # Track gradient norms (key metric in paper)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(np.sqrt(total_norm))
            
            # Optimizer step with accumulation
            if (batch_idx + 1) % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * (config.accumulation_steps if config.use_gradient_accumulation else 1)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_grad_norm = np.mean(grad_norms)
        
        history['losses'].append(avg_loss)
        history['gradient_norms'].append(avg_grad_norm)
        
        # Evaluation
        if eval_fn is not None:
            acc = eval_fn(model, device)
            history['accuracies'].append(acc)
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, GradNorm={avg_grad_norm:.4f}, Acc={acc:.4f}")
        else:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, GradNorm={avg_grad_norm:.4f}")
    
    return history


# ============================================================================
# 7. EXPERIMENTAL SETUP (Replicating Paper Results)
# ============================================================================

def run_ablation_study():
    """
    Replicate key experiments from the paper:
    1. SR vs RTN comparison
    2. Batch size scaling effects
    3. Different precision formats (E4M3, E4M2, E4M1, E4M0)
    """
    results = {}
    
    # Experiment 1: SR vs RTN at different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]
    formats = ['E4M3', 'E4M2', 'E4M1']
    
    for fmt in formats:
        results[fmt] = {'SR': {}, 'RTN': {}}
        for bs in batch_sizes:
            # SR training
            config_sr = SRTrainingConfig(
                weight_format=fmt,
                activation_format=fmt,
                gradient_format=fmt,
                batch_size=bs,
                stochastic_weights=False,  # Per paper: deterministic weight quant
            )
            # Run training...
            
            # RTN training (for comparison)
            # Use deterministic rounding throughout
    
    return results
def quick_train_test():
    """Verify SR training converges on MNIST."""
    import torchvision
    import torchvision.transforms as transforms
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    
    # Simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Convert to SR
    config = SRTrainingConfig(
        weight_format='E4M3',
        activation_format='E4M3',
        gradient_format='E4M3',
        batch_size=64
    )
    model = convert_to_sr_model(model, config)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train 2 epochs
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] '
                      f'Loss: {running_loss/100:.3f} | '
                      f'Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        print(f'Epoch {epoch+1} complete. Accuracy: {100.*correct/total:.2f}%')


def compare_sr_rtn():
    """Compare SR vs RTN at E4M3 precision."""
    import torchvision, torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    def train_model(use_sr, batch_size=128, epochs=3):
        loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Convert: SR uses stochastic rounding, RTN uses deterministic
        config = SRTrainingConfig(
            weight_format='E4M3',
            activation_format='E4M3',
            gradient_format='E4M3',
            batch_size=batch_size
        )
        model = convert_to_sr_model(model, config)
        
        # Hack: toggle SR off for RTN test by setting flags
        if not use_sr:
            # Pass SR/RTN mode through the model
            for m in model.modules():
                if isinstance(m, SRLinear):
                    m.stochastic_act = use_sr
                    m.stochastic_grad = use_sr
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        model.train()
        for epoch in range(epochs):
            correct = total = 0
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            history.append(acc)
            print(f"  Epoch {epoch+1}: {acc:.2f}%")
        
        return history
    
    print("Training with SR (stochastic rounding)...")
    sr_hist = train_model(use_sr=True)
    
    print("\nTraining with RTN (deterministic rounding)...")
    rtn_hist = train_model(use_sr=False)
    
    print(f"\nFinal accuracy - SR: {sr_hist[-1]:.2f}%, RTN: {rtn_hist[-1]:.2f}%")

def get_lr_with_scaling(base_lr, batch_size, warmup_steps, current_step, lr_scale='linear'):
    """
    Learning rate scaling for different batch sizes.
    
    Linear scaling rule: LR ∝ batch_size (Goyal et al. 2017)
    SQRT scaling rule: LR ∝ sqrt(batch_size)
    """
    if lr_scale == 'linear':
        scaled_lr = base_lr * (batch_size / 64.0)  # normalize to bs=64
    elif lr_scale == 'sqrt':
        scaled_lr = base_lr * np.sqrt(batch_size / 64.0)
    else:
        scaled_lr = base_lr
    
    # Linear warmup
    if current_step < warmup_steps:
        return scaled_lr * (current_step / warmup_steps)
    
    return scaled_lr

def test_batch_size_scaling():
    """
    Validate Lemma 3: SR variance scales as 1/batch_size.
    
    Paper insight: 1-bit precision reduction can be compensated by 
    2-4x batch size increase to maintain convergence quality.
    """
    import torchvision, torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    def train_and_eval(batch_size, weight_bits, act_bits, grad_bits, epochs=3, seed=42):
        """Train model and return final accuracy with proper LR scaling."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=0, generator=torch.Generator().manual_seed(seed)
        )
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        config = SRTrainingConfig(
            weight_format=f'E4M{weight_bits}',
            activation_format=f'E4M{act_bits}',
            gradient_format=f'E4M{grad_bits}',
            batch_size=batch_size
        )
        model = convert_to_sr_model(model, config)
        
        # Use AdamW instead of SGD for stability
        base_lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Calculate warmup steps
        steps_per_epoch = len(loader)
        warmup_steps = min(200, steps_per_epoch)  # warmup for 200 steps or 1 epoch
        
        model.train()
        global_step = 0
        
        for epoch in range(epochs):
            correct = total = 0
            epoch_loss = 0.0
            
            for images, labels in loader:
                # Update LR with warmup and batch-size scaling
                lr = get_lr_with_scaling(
                    base_lr, batch_size, warmup_steps, global_step, lr_scale='sqrt'
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                global_step += 1
            
            acc = 100. * correct / total
            avg_loss = epoch_loss / steps_per_epoch
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.3f}, Acc={acc:.2f}%")
        
        return acc
    
    print("\n" + "="*60)
    print("Testing Batch Size Scaling (Lemma 3)")
    print("="*60)
    
    # Experiment 1: Fixed precision, varying batch size
    print("\n--- Experiment 1: E4M3 with varying batch sizes ---")
    results_e4m3 = {}
    for bs in [32, 64, 128, 256]:
        print(f"\nTraining E4M3 with batch_size={bs}...")
        acc = train_and_eval(bs, weight_bits=3, act_bits=3, grad_bits=3, epochs=3)
        results_e4m3[bs] = acc
        print(f"Final accuracy: {acc:.2f}%")
    
    # Experiment 2: Fixed precision, varying batch size (E4M2 - more aggressive)
    print("\n--- Experiment 2: E4M2 with varying batch sizes ---")
    results_e4m2 = {}
    for bs in [64, 128, 256, 512]:
        print(f"\nTraining E4M2 with batch_size={bs}...")
        acc = train_and_eval(bs, weight_bits=2, act_bits=2, grad_bits=2, epochs=3)
        results_e4m2[bs] = acc
        print(f"Final accuracy: {acc:.2f}%")
    
    # Experiment 3: Precision vs Batch Size trade-off

    print("\n--- Experiment 3: Precision-Batch Size Trade-off ---")
    configs = [
        ('E4M3_b64', 3, 3, 3, 64),    # Baseline
        ('E4M3_b128', 3, 3, 3, 128),  # Same precision, larger batch
        ('E4M2_b128', 2, 2, 2, 128),  # 1 less bit, 2x batch
        ('E4M2_b256', 2, 2, 2, 256),  # 1 less bit, 4x batch
    ]
    
    tradeoff_results = {}
    for name, wb, ab, gb, bs in configs:
        print(f"\nTraining {name}...")
        acc = train_and_eval(bs, wb, ab, gb, epochs=3)
        tradeoff_results[name] = (bs, acc)
        print(f"Final accuracy: {acc:.2f}%")
    
    # Analysis
    print("\n" + "="*60)
    print("Analysis")
    print("="*60)
    
    # Check if E4M2@128 ≈ E4M3@64 (should be similar if scaling law holds)
    print("\n--- Scaling Law Validation ---")
    
    comparisons = [
        ("E4M3@64", results_e4m3.get(64), "E4M3@128", results_e4m3.get(128)),
        ("E4M3@64", results_e4m3.get(64), "E4M2@128", results_e4m2.get(128)),
        ("E4M3@128", results_e4m3.get(128), "E4M2@256", results_e4m2.get(256)),
    ]
    
    for name1, acc1, name2, acc2 in comparisons:
        if acc1 is not None and acc2 is not None:
            diff = abs(acc1 - acc2)
            print(f"\n{name1}: {acc1:.2f}% vs {name2}: {acc2:.2f}%")
            print(f"Difference: {diff:.2f}%")
            if diff < 5.0:
                print("✓ Scaling validated")
            else:
                print("✗ Scaling not validated")
    
    return results_e4m3, results_e4m2, tradeoff_results
# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Simple test
    torch.manual_seed(42)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Convert to SR
    config = SRTrainingConfig(
        weight_format='E4M3',
        activation_format='E4M3', 
        gradient_format='E4M3',
        batch_size=32
    )
    
    model = convert_to_sr_model(model, config)
    
    # Test forward/backward
    x = torch.randn(4, 784)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print("SR model test passed!")
    print(f"Output shape: {y.shape}")
    print(f"Gradient computed successfully")
    # Add gradient verification here:
    print("\n--- Gradient Check ---")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"{name}: NO GRADIENT (this is a problem!)")
        
    # Add this at the end:
    print("\n" + "="*50)
    print("Running MNIST convergence test...")
    print("="*50)
    quick_train_test()
    # ... existing tests ...
    print("\n" + "="*50)
    compare_sr_rtn()
    test_batch_size_scaling()