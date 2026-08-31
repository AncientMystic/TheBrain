
import sys
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core import db

class ExtractionDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_len=512, max_target_len=2048):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        input_text = "Convert the following text into structured JSON: " + self.inputs[idx][:1000]
        target_text = self.targets[idx]
        src = self.tokenizer(input_text, truncation=True, max_length=self.max_input_len, return_tensors="pt", padding="max_length")
        tgt = self.tokenizer(target_text, truncation=True, max_length=self.max_target_len, return_tensors="pt", padding="max_length")
        # Replace padding token ids with -100 so loss ignores them
        labels = tgt["input_ids"].squeeze()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        labels[labels == pad_token_id] = -100
        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels,
        }

def main():
    parser = argparse.ArgumentParser(description="Train distilled extractor.")
    parser.add_argument("--epoch", type=int, default=None, help="Number of epochs.")
    args = parser.parse_args()

    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT chunk_text, target_json FROM distilled_training_data")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("No training data. Run guided learning first.")
        return

    inputs = [r["chunk_text"] for r in rows]
    targets = [r["target_json"] for r in rows]

    model_dir = Path(config.DISTILLED_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.DISTILLED_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(config.DISTILLED_MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    dataset = ExtractionDataset(inputs, targets, tokenizer)
    # Simple train/val split
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    epochs = args.epoch if args.epoch is not None else getattr(config, "DISTILLED_TRAINING_EPOCHS", 5)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    print(f"Best validation loss: {best_val_loss:.4f}. Model saved to {model_dir}")

if __name__ == "__main__":
    main()
