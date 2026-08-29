
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core.embeddings import get_embeddings_batch
from core.hyperbolic import exp_map, hyperbolic_distance

def collect_positive_negative_pairs():
    """Use verified facts to build positive/negative pairs."""
    from core import db
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT fact_text, canonical_value FROM key_facts WHERE verification_status='verified' LIMIT 100")
    rows = cur.fetchall()
    conn.close()
    texts = [r["fact_text"] for r in rows]
    embs = get_embeddings_batch(texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    # Positive: facts with same canonical_value; negative: different
    pairs = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if rows[i]["canonical_value"] == rows[j]["canonical_value"]:
                pairs.append((embs[i], embs[j], 1))
            else:
                pairs.append((embs[i], embs[j], 0))
            if len(pairs) > 500:
                break
        if len(pairs) > 500:
            break
    return pairs

def train_projection():
    pairs = collect_positive_negative_pairs()
    if not pairs:
        print("No verified facts to train contrastive model.")
        return
    # Simple linear projection to hyperbolic space trained with contrastive loss
    dim = len(pairs[0][0])
    proj = nn.Linear(dim, 64)
    optimizer = torch.optim.Adam(proj.parameters(), lr=0.001)
    criterion = nn.MarginRankingLoss(margin=1.0)
    for epoch in range(20):
        total_loss = 0
        for e1, e2, label in pairs:
            e1 = torch.tensor(exp_map(e1), dtype=torch.float32)
            e2 = torch.tensor(exp_map(e2), dtype=torch.float32)
            z1 = proj(e1)
            z2 = proj(e2)
            d = torch.norm(z1 - z2, dim=1)
            if label == 1:
                loss = criterion(-d, torch.zeros_like(d), torch.ones_like(d))
            else:
                loss = criterion(d, torch.ones_like(d), torch.ones_like(d))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(pairs):.4f}")
    torch.save(proj.state_dict(), Path(config.BASE_DIR) / "models" / "contrastive_proj.pt")
    print("Saved contrastive projection.")

if __name__ == "__main__":
    train_projection()
