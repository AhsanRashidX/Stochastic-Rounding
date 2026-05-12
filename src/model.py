

# gpt_experiment.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import math
import json

from quantized_layers import replace_linear_with_quantized
from train_utils import train, evaluate

# ============== SETUP ==============
model_name = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    texts = [t for t in examples["text"] if len(t.strip()) > 0]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Use more data for better results
small_train = tokenized["train"].select(range(20000))  # Increased
small_val = tokenized["validation"].select(range(2000))

print(f"Dataset ready. Device: {device}")

# ============== EXPERIMENT CONFIGURATIONS ==============
configs = [
    {"name": "FP32_baseline", "bits": 32, "batch_size": 8, "epochs": 10, "lr": 5e-5},
    {"name": "SR_8bit_b8", "bits": 8, "batch_size": 8, "epochs": 10, "lr": 5e-5},
    {"name": "RTN_8bit_b8", "bits": 8, "batch_size": 8, "epochs": 10, "lr": 5e-5, "use_sr": False},
    {"name": "SR_8bit_b32", "bits": 8, "batch_size": 32, "epochs": 10, "lr": 5e-5},
    {"name": "SR_6bit_b8", "bits": 6, "batch_size": 8, "epochs": 10, "lr": 5e-5},
    {"name": "SR_6bit_b32", "bits": 6, "batch_size": 32, "epochs": 10, "lr": 5e-5},
    {"name": "RTN_6bit_b32", "bits": 6, "batch_size": 32, "epochs": 10, "lr": 5e-5, "use_sr": False},
    {"name": "SR_4bit_b8", "bits": 4, "batch_size": 8, "epochs": 10, "lr": 2e-5},
    {"name": "SR_4bit_b32", "bits": 4, "batch_size": 32, "epochs": 10, "lr": 2e-5},
    {"name": "RTN_4bit_b32", "bits": 4, "batch_size": 32, "epochs": 10, "lr": 2e-5, "use_sr": False},
]

results = []

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Experiment: {cfg['name']}")
    print(f"Bits: {cfg['bits']}, Batch: {cfg['batch_size']}, LR: {cfg['lr']}")
    print('='*60)
    
    train_loader = DataLoader(small_train, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(small_val, batch_size=cfg['batch_size'])
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if cfg['bits'] < 32:
        use_sr = cfg.get('use_sr', True)
        print(f"Applying {cfg['bits']}-bit quantization ({'SR' if use_sr else 'RTN'})...")
        replace_linear_with_quantized(model, bits=cfg['bits'], use_sr=use_sr)
    else:
        print("Running FP32 baseline...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    model.to(device)
    
    # optimizer = AdamW(model.parameters(), lr=cfg['lr'])
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    
    best_val_loss = float('inf')
    best_ppl = float('inf')
    
    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, ppl = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ppl = ppl
    
    results.append({
        'name': cfg['name'],
        'bits': cfg['bits'],
        'batch_size': cfg['batch_size'],
        'best_val_loss': best_val_loss,
        'best_ppl': best_ppl
    })
    
    print(f"\nBest for {cfg['name']}: Loss={best_val_loss:.4f}, PPL={best_ppl:.2f}")

# ============== FINAL RESULTS ==============
print(f"\n{'='*80}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*80}")
print(f"{'Experiment':<20} | {'Bits':<6} | {'Batch':<6} | {'Val Loss':<10} | {'Perplexity':<12}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<20} | {r['bits']:<6} | {r['batch_size']:<6} | {r['best_val_loss']:<10.4f} | {r['best_ppl']:<12.2f}")

print(f"{'='*80}")

with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to experiment_results.json")
