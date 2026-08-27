#!/usr/bin/env python3
"""Train topic shift LSTM on synthetic data."""
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import db
from core.local_embedder import get_local_embedder
from core.topic_shift_model import TopicShiftLSTM

def generate_synthetic_data(num_samples=5000):
    embedder = get_local_embedder()
    if not embedder.available:
        raise RuntimeError("Local embedder not available for training.")

    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT fact_id, fact_text, canonical_value FROM key_facts LIMIT 2000")
    rows = cur.fetchall()
    conn.close()
    if len(rows) < 10:
        raise RuntimeError("Not enough facts.")
    facts = [dict(r) for r in rows]

    conversations = []
    for _ in range(num_samples):
        f1 = random.choice(facts)
        topic1 = f"Tell me about {f1['canonical_value'] or f1['fact_text'][:50]}"
        if random.random() < 0.5:
            topic2 = f"What else is related to {f1['canonical_value'] or f1['fact_text'][:50]}?"
            label = 0
        else:
            f2 = random.choice(facts)
            topic2 = f"Tell me about {f2['canonical_value'] or f2['fact_text'][:50]}"
            label = 1
        conversations.append(([topic1], topic2, label))

    X = []
    y = []
    entity_features = []
    for history, query, label in conversations:
        emb_hist = embedder.encode([history[0]])[0]
        emb_query = embedder.encode([query])[0]
        seq = np.stack([emb_hist, emb_query])
        X.append(seq)
        entity_features.append(np.zeros(10))
        y.append(label)
    return np.array(X), np.array(entity_features), np.array(y, dtype=np.float32)

def train():
    X, entity_features, y = generate_synthetic_data(1000)
    input_dim = X.shape[2]
    model = TopicShiftLSTM(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(entity_features, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_ef, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_ef)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {total_loss/len(loader):.4f}")

    out_dir = Path(config.BASE_DIR) / "models" / "topic_shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "topic_shift_lstm.pt")
    print("Model saved.")

if __name__ == "__main__":
    train()
