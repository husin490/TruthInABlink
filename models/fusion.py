"""
Truth in a Blink — Attention-Based Fusion Module
==================================================
Combines macro (facial context) and micro (motion) stream embeddings
via learned gated fusion, producing:
  • A fused representation (256-dim)
  • Interpretable attention weights  w_macro, w_micro  with  w_macro + w_micro = 1

Design
------
Both embeddings are projected into a shared space. A gating network
computes attention logits from the concatenated representation.
A softmax ensures the constraint w_macro + w_micro = 1.

The fused result is further refined through a non-linear transformation
to capture cross-stream interactions.

Note: Uses element-wise operations throughout to avoid small-tensor
matmul issues on Apple Silicon MPS backend (PyTorch 2.8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Dual-stream gated attention fusion (MPS-compatible).

    Parameters
    ----------
    macro_dim  : Dimensionality of macro-stream embedding.
    micro_dim  : Dimensionality of micro-stream embedding.
    hidden_dim : Internal hidden dimensionality.
    num_heads  : Kept for config compatibility (not used).
    dropout    : Dropout probability.
    """

    def __init__(
        self,
        macro_dim: int = 256,
        micro_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project both streams to shared space
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.micro_proj = nn.Linear(micro_dim, hidden_dim)

        # ── Gated attention network ──────────────────────────────────────
        # Computes w_macro, w_micro from the joint representation.
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Weight predictor → 2 logits → softmax → w_macro, w_micro
        self.weight_head = nn.Linear(hidden_dim, 2)

        # ── Cross-interaction layer ──────────────────────────────────────
        # Captures non-linear interactions between streams
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        macro_emb: torch.Tensor,
        micro_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        macro_emb : (B, macro_dim)
        micro_emb : (B, micro_dim)

        Returns
        -------
        fused     : (B, hidden_dim)  — fused representation.
        w_macro   : (B, 1)           — macro attention weight.
        w_micro   : (B, 1)           — micro attention weight.
        """
        # Project both streams to shared space
        macro_h = self.macro_proj(macro_emb)              # (B, D)
        micro_h = self.micro_proj(micro_emb)              # (B, D)

        # Concatenate for joint reasoning
        concat = torch.cat([macro_h, micro_h], dim=-1)    # (B, 2*D)

        # ── Gated attention weights ──────────────────────────────────────
        gate_features = self.gate_net(concat)              # (B, D)
        weight_logits = self.weight_head(gate_features)    # (B, 2)
        weights = F.softmax(weight_logits, dim=-1)         # w_macro + w_micro = 1
        w_macro = weights[:, 0:1]                          # (B, 1)
        w_micro = weights[:, 1:2]                          # (B, 1)

        # ── Weighted combination ─────────────────────────────────────────
        fused_weighted = w_macro * macro_h + w_micro * micro_h  # (B, D)

        # ── Cross-interaction ────────────────────────────────────────────
        interaction = self.interaction(concat)              # (B, D)

        # Combine weighted fusion with cross-interaction
        fused = self.norm(
            self.dropout_layer(fused_weighted + interaction)
        )  # (B, D)

        return fused, w_macro, w_micro
