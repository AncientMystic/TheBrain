
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core import db

class ExtractionDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_len=512, max_target_len=1024):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        input_text = "extract: " + self.inputs[idx][:1000]
        target_text = self.targets[idx]
        src = self.tokenizer(input_text, truncation=True, max_length=self.max_input_len, return_tensors="pt", padding="max_length")
        tgt = self.tokenizer(target_text, truncation=True, max_length=self.max_target_len, return_tensors="pt", padding="max_length")
        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": tgt["input_ids"].squeeze(),
        }

def main():
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT chunk_text, target_json FROM distilled_training_data")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("No training data. Run guided learning with LLM first.")
        return
    inputs = [r["chunk_text"] for r in rows]
    targets = [r["target_json"] for r in rows]

    model_dir = Path(config.DISTILLED_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.DISTILLED_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.DISTILLED_MODEL_NAME)

    dataset = ExtractionDataset(inputs, targets, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = getattr(config, "DISTILLED_TRAINING_EPOCHS", 3)
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss: {total_loss/len(train_loader):.4f}")

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Distilled extractor saved to {model_dir}")

if __name__ == "__main__":
    main()
