"""
Contextual topic shift detection using a learned sequence model.
Falls back to heuristic if model not available.
"""
import numpy as np
import torch
import torch.nn as nn
import config
from pathlib import Path
from core.local_embedder import get_local_embedder


class TopicShiftLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim + 10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, entity_features):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        combined = torch.cat([last_out, entity_features], dim=1)
        logits = self.fc(combined)
        return self.sigmoid(logits).squeeze(-1)


class TopicShiftModelDetector:
    def __init__(self, model_dir=None, sequence_length=4):
        self.sequence_length = sequence_length
        self.model = None
        self.embedder = get_local_embedder()
        if not self.embedder.available:
            from core.embeddings import get_embedding
            self.embed_fn = get_embedding
        else:
            self.embed_fn = self.embedder.encode

        if model_dir is None:
            model_dir = Path(config.BASE_DIR) / "models" / "topic_shift"
        self.model_dir = model_dir
        self._load_model()

    def _load_model(self):
        model_path = self.model_dir / "topic_shift_lstm.pt"
        if model_path.exists():
            try:
                input_dim = 384  # MiniLM dimension
                self.model = TopicShiftLSTM(input_dim)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                print("Topic shift LSTM loaded.")
            except Exception as e:
                print(f"Failed to load topic shift model: {e}")
                self.model = None

    def _get_embedding(self, text):
        if callable(self.embed_fn):
            if self.embedder.available:
                return self.embedder.encode([text])[0]
            else:
                return self.embed_fn(text)
        return None

    def _extract_entity_overlap(self, query, history):
        from chat.query_analyzer import extract_topic_terms
        q_entities = set(extract_topic_terms(query))
        if not q_entities:
            return [0.0] * 10
        features = []
        for past_query in history[-10:]:
            past_entities = set(extract_topic_terms(past_query))
            overlap = len(q_entities & past_entities) / max(1, len(q_entities))
            features.append(overlap)
        if len(features) < 10:
            features.extend([0.0] * (10 - len(features)))
        return features[:10]

    def is_new_topic(self, query, history):
        if self.model is None:
            if not history:
                return False
            from chat.query_analyzer import extract_topic_terms
            prev_entities = set(extract_topic_terms(history[-1]))
            curr_entities = set(extract_topic_terms(query))
            if not curr_entities:
                return False
            overlap = len(prev_entities & curr_entities) / max(1, len(curr_entities))
            return overlap < 0.3

        seq_queries = history[-(self.sequence_length-1):] + [query] if history else [query]
        embeddings = []
        for q in seq_queries:
            emb = self._get_embedding(q)
            if emb is not None:
                embeddings.append(emb)
            else:
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[-1]))
                else:
                    return False
        while len(embeddings) < self.sequence_length:
            embeddings.insert(0, np.zeros_like(embeddings[-1]))
        x = np.stack(embeddings)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        entity_feats = self._extract_entity_overlap(query, history)
        entity_feats = torch.tensor(entity_feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(x, entity_feats).item()
        return prob > 0.5
