# debug_quantization.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from quantized_layers import replace_linear_with_quantized

model_name = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("DEBUG: Checking if quantization is actually applied")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(model_name)

# Count ALL linear layers (including nested)
linear_count = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
print(f"\nTotal Linear layers in model: {linear_count}")

# Apply 4-bit quantization (quantize everything including lm_head for testing)
print("\n--- Applying 4-bit quantization ---")
replace_linear_with_quantized(model, bits=4, use_sr=True, skip_lm_head=False)

# Count quantized layers
q_count = sum(1 for _, m in model.named_modules() if "QuantizedLinearLayer" in type(m).__name__)
print(f"\nTotal QuantizedLinear layers: {q_count}")

# Test forward + backward
model.to(device)
model.train()
dummy_input = torch.randint(0, 50257, (2, 32)).to(device)
attention_mask = torch.ones_like(dummy_input)

print("\nRunning forward + backward...")
outputs = model(dummy_input, attention_mask=attention_mask, labels=dummy_input)
loss = outputs.loss
loss.backward()

print(f"\nLoss: {loss.item():.4f}")
print(f"\nExpected: {q_count} quantized layers (was {linear_count} linear)")
print("If q_count << linear_count, the recursion is broken!")