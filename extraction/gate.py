
import numpy as np
import json
from pathlib import Path
import config

PRIME_INDICES = {2, 3, 5, 7, 11, 13, 17, 19}  # 1-indexed prime numbers up to 22
EVEN_INDICES = set(range(2, 23, 2))  # 1-indexed even numbers

class PrimeEvenGate:
    def __init__(self):
        # Parameter shapes:
        # beta: 23 = intercept + 22 loadings (top-band gate)
        # gamma: 22 = intercept + 21 loadings (gap gate, used only for regularization)
        # delta: 23 = intercept + 22 loadings (unitary-coupled gate, anchor regularized)
        self.beta = np.zeros(23, dtype=np.float32)
        self.gamma = np.zeros(22, dtype=np.float32)
        self.delta = np.zeros(23, dtype=np.float32)
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(42)
        self.beta = rng.normal(0, 0.1, self.beta.shape).astype(np.float32)
        self.gamma = rng.normal(0, 0.1, self.gamma.shape).astype(np.float32)
        self.delta = rng.normal(0, 0.1, self.delta.shape).astype(np.float32)

    def forward(self, features):
        """
        features: np.array of length 44 = [22 singular values, 21 gaps, 1 phase]
        Returns gate weight w_R(x) in [0,1] using the top-band gate.
        """
        features = np.asarray(features, dtype=np.float32)
        # Top-band gate: w_top = sigma(beta0 + sum_{i=1..22} beta_i * lambda_i)
        top_input = self.beta[0] + np.dot(self.beta[1:], features[:22])
        w = 1.0 / (1.0 + np.exp(-top_input))
        return float(np.clip(w, 0.0, 1.0))

    def _soft_threshold(self, x, threshold):
        """Element-wise soft-thresholding."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _prox_update_beta(self, grad, lr, lam1, lam2):
        """
        Update beta[1:] with proximal gradient.
        For prime indices: only L1 (lam1)
        For off-prime indices: L1 (lam1) + prime-pull penalty (lam2)
        """
        theta = self.beta[1:] - lr * grad
        thresholds = np.full(22, lr * lam1, dtype=np.float32)
        # Add extra threshold for off-prime entries
        off_prime_mask = np.ones(22, dtype=bool)
        for idx in PRIME_INDICES:
            off_prime_mask[idx-1] = False  # keep prime indices untouched by extra
        thresholds[off_prime_mask] += lr * lam2
        self.beta[1:] = self._soft_threshold(theta, thresholds)

    def _prox_update_gamma(self, grad, lr, lam1, lam3):
        """
        Update gamma[1:] with proximal gradient.
        For even indices: only L1 (lam1)
        For off-even indices: L1 (lam1) + even-pull penalty (lam3)
        """
        theta = self.gamma[1:] - lr * grad
        thresholds = np.full(21, lr * lam1, dtype=np.float32)
        off_even_mask = np.ones(21, dtype=bool)
        for idx in EVEN_INDICES:
            if idx <= 21:
                off_even_mask[idx-1] = False
        thresholds[off_even_mask] += lr * lam3
        self.gamma[1:] = self._soft_threshold(theta, thresholds)

    def _prox_update_delta(self, grad, lr, lam1, lam4):
        """
        Update delta[1:] with L1 and anchor penalty.
        Anchor: |delta[2] - mean(delta[prime_indices])|
        """
        theta = self.delta[1:] - lr * grad
        # L1 threshold for all loadings
        thresholds = np.full(22, lr * lam1, dtype=np.float32)
        self.delta[1:] = self._soft_threshold(theta, thresholds)

        # Anchor penalty gradient (subgradient)
        # Identify 0-indexed positions for prime indices and anchor index 2 (0-indexed)
        prime_positions = [idx-1 for idx in PRIME_INDICES if idx <= 22]
        anchor_pos = 1  # delta[2] is index 2 in full array -> index 1 in loadings (0-indexed)
        # current values
        anchor_val = self.delta[2]  # full array index 2
        prime_vals = [self.delta[idx] for idx in PRIME_INDICES if idx <= len(self.delta)]
        avg_prime = np.mean(prime_vals) if prime_vals else 0.0
        diff = anchor_val - avg_prime
        sign = np.sign(diff) if diff != 0 else 0.0

        # Apply gradient: sign for anchor, -sign/len(primes) for each prime
        self.delta[2] -= lr * lam4 * sign
        for idx in PRIME_INDICES:
            if idx <= len(self.delta):
                self.delta[idx] += lr * lam4 * sign / len(prime_positions)

    def train_step(self, features, labels, lr=0.01, lam1=0.01, lam2=0.1, lam3=0.1, lam4=0.05):
        """
        One step of proximal gradient descent on all parameters.
        features: np.array shape (batch, 44)
        labels: np.array shape (batch,) 0/1
        """
        batch_size = len(labels)
        if batch_size == 0:
            return 0.0

        # Forward pass
        outputs = np.array([self.forward(f) for f in features], dtype=np.float32)
        eps = 1e-8
        # BCE gradient wrt gate output
        dL_dw = -(labels / (outputs + eps) - (1 - labels) / (1 - outputs + eps)) / batch_size

        # Since w = top-band gate only, gradients only affect beta.
        # For completeness, we still compute dummy gradients for gamma/delta (zero from data loss)
        grad_beta = np.zeros_like(self.beta)
        grad_beta[0] = np.sum(dL_dw)
        for i in range(22):
            grad_beta[i+1] = np.sum(dL_dw * features[:, i])

        # gamma/delta get no data gradient (their regularizer will still update)
        grad_gamma = np.zeros_like(self.gamma)
        grad_delta = np.zeros_like(self.delta)

        # Update intercepts without regularization
        self.beta[0] -= lr * grad_beta[0]
        self.gamma[0] -= lr * grad_gamma[0]
        self.delta[0] -= lr * grad_delta[0]

        # Update loadings with proximal operators
        self._prox_update_beta(grad_beta[1:], lr, lam1, lam2)
        self._prox_update_gamma(grad_gamma[1:], lr, lam1, lam3)
        self._prox_update_delta(grad_delta[1:], lr, lam1, lam4)

        # Compute loss for monitoring
        loss = -np.mean(labels * np.log(outputs + eps) + (1 - labels) * np.log(1 - outputs + eps))
        return loss

    def save(self, path):
        data = {
            'beta': self.beta.tolist(),
            'gamma': self.gamma.tolist(),
            'delta': self.delta.tolist(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.beta = np.array(data['beta'], dtype=np.float32)
        self.gamma = np.array(data['gamma'], dtype=np.float32)
        self.delta = np.array(data['delta'], dtype=np.float32)
