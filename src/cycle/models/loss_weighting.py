"""Learned homoscedastic-uncertainty loss weights (Kendall et al., 2018).

EXTENSION (not in the paper): replaces the fixed Eq. 15 weights a, b, c with
    L = sum_i  L_i / (2 sigma_i^2) + log sigma_i
with one learned log sigma_i per term. For fixed L_i the optimum is
sigma_i^2 = L_i, so the effective weight 1 / (2 sigma_i^2) tends to 1 / (2 L_i).

log sigma_i is stored as s0_i + delta_i, with s0_i = -log(2 w_i) / 2 fixed and
delta_i learned from 0, so the effective weight w_i * exp(-2 delta_i) equals the
fixed weight w_i exactly at step 0 (same objective and gradients as learning
log sigma_i directly).
"""
import math

import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    def __init__(self, names, init_weights):
        super().__init__()
        if any(w <= 0 for w in init_weights):
            raise ValueError(f"learned weights need positive initial weights, got {init_weights}")
        self.names = list(names)
        self.register_buffer("w0", torch.tensor(init_weights, dtype=torch.float32))
        self.register_buffer("s0", torch.tensor(
            [-0.5 * math.log(2.0 * w) for w in init_weights], dtype=torch.float32))
        self.delta = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, losses):
        """losses: name -> scalar tensor, for the names given at init.
        Returns (weighted sum of the losses, regularizer sum of log sigma)."""
        w = self.w0 * torch.exp(-2.0 * self.delta)
        weighted = sum(w[i] * losses[n] for i, n in enumerate(self.names))
        return weighted, (self.s0 + self.delta).sum()

    def weights(self):
        """Current effective weights 1 / (2 sigma^2), by name."""
        w = self.w0 * torch.exp(-2.0 * self.delta.detach())
        return {n: w[i].item() for i, n in enumerate(self.names)}
