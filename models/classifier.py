"""
Truth in a Blink — Deception Classifier Head
==============================================
Binary classification: Truthful (0) vs Deceptive (1).
Outputs probability of deception ∈ [0, 1].
"""

import torch
import torch.nn as nn


class DeceptionClassifier(nn.Module):
    """
    Two-layer MLP head for binary deception classification.

    Parameters
    ----------
    input_dim  : Fused embedding dimension (default 256).
    hidden_dim : Hidden layer width.
    dropout    : Dropout probability.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),       # single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim) — fused embedding.

        Returns
        -------
        prob : (B, 1) — deception probability after sigmoid.
        """
        return torch.sigmoid(self.net(x))
