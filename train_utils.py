# train_utils.py
import torch
import math

def train(model, loader, optimizer, device, max_grad_norm=1.0):
    """Training with gradient clipping for quantized training stability"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (essential for quantized training)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(loader) if len(loader) > 0 else 0


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity