
import numpy as np
import json
from pathlib import Path
import config

VERIFIER_NAMES = ["text_grounding", "symstep", "vericot", "fidelis", "rcot", "ares"]

class VerificationGate:
    """Outputs per-verifier weights based on spectral features of a claim."""
    def __init__(self, feat_dim=44, num_verifiers=6):
        self.feat_dim = feat_dim
        self.num_verifiers = num_verifiers
        # Weight matrix: num_verifiers x (feat_dim+1)
        self.W = np.zeros((num_verifiers, feat_dim + 1), dtype=np.float32)
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.1, self.W.shape).astype(np.float32)

    def forward(self, features):
        """Return dict of verifier_name -> weight in [0,1]."""
        features = np.asarray(features, dtype=np.float32)
        if len(features) < self.feat_dim:
            features = np.pad(features, (0, self.feat_dim - len(features)))
        else:
            features = features[:self.feat_dim]
        x = np.concatenate([[1.0], features])  # add bias
        logits = self.W @ x
        weights = 1.0 / (1.0 + np.exp(-logits))  # sigmoid per verifier
        return {name: float(w) for name, w in zip(VERIFIER_NAMES, weights)}

    def save(self, path):
        data = {'W': self.W.tolist()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.W = np.array(data['W'], dtype=np.float32)
