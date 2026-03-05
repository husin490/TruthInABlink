"""
Truth in a Blink — Macro Stream (Vision Transformer)
=====================================================
A compact ViT backbone pretrained on FER2013 for facial-context embeddings.

Architecture
------------
1.  Patch embedding  (224×224 → 14×14 = 196 patches of dim 384)
2.  Learnable [CLS] token + positional embedding
3.  L transformer encoder blocks
4.  CLS token → projection → 256-dim embedding

During FER2013 pretraining an emotion-classification head is attached.
After pretraining the head is removed and the backbone serves as a
frozen / fine-tunable feature extractor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ─────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and linearly embed them."""

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional dropout."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).contiguous().reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, B, heads, N, d_k)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─── Macro Stream ViT ────────────────────────────────────────────────────────

class MacroStreamViT(nn.Module):
    """
    Vision Transformer backbone for the macro (facial context) stream.

    Operates in two modes:

    **Pretraining mode** (default at construction)
        ``fer_head`` is present; ``forward()`` returns emotion logits.
        Used during Stage 1 FER2013 training.

    **Feature-extractor mode** (after ``to_feature_extractor()``)
        ``fer_head`` is deleted; ``forward()`` returns a 256-dim embedding
        identical to ``forward_features()``.  Used during Stage 2+ RLDD.

    Parameters
    ----------
    image_size      : Input resolution (default 224).
    patch_size      : Patch side (default 16).
    in_channels     : Image channels (default 3).
    embed_dim       : Transformer hidden dimension.
    depth           : Number of transformer blocks.
    num_heads       : Attention heads per block.
    mlp_ratio       : MLP expansion ratio.
    dropout         : Dropout probability.
    output_dim      : Dimensionality of the output embedding.
    num_fer_classes : Number of FER2013 emotion classes (pretraining head).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 256,
        num_fer_classes: int = 7,
    ):
        super().__init__()

        self._feature_extractor_mode = False

        # Patch + positional embeddings
        self.patch_embed = PatchEmbedding(image_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Projection to output_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
        )

        # FER2013 pretraining classification head (detachable)
        self.fer_head = nn.Linear(output_dim, num_fer_classes)

        self._init_weights()

    # ── Weight initialisation ────────────────────────────────────────────
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Mode switching ───────────────────────────────────────────────────

    def to_feature_extractor(self):
        """
        Strip the emotion classification head and switch to
        feature-extractor mode.  After this call:
          • ``fer_head`` is removed (saves memory/params)
          • ``forward()`` returns 256-dim embeddings, not logits
        """
        if hasattr(self, "fer_head"):
            del self.fer_head
        self._feature_extractor_mode = True
        print("[✓] Macro stream converted to feature-extractor mode "
              "(fer_head removed).")
        return self

    @property
    def is_feature_extractor(self) -> bool:
        return self._feature_extractor_mode

    # ── Freezing helpers ─────────────────────────────────────────────────

    def freeze_backbone(self, keep_projection_trainable: bool = True):
        """
        Freeze the transformer encoder (patches, pos embed, blocks, norm)
        while optionally keeping the projection layer trainable so it can
        adapt to the downstream deception task.
        """
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        # Selectively unfreeze projection
        if keep_projection_trainable:
            for p in self.projection.parameters():
                p.requires_grad = True

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"[✓] Macro backbone frozen  "
              f"(frozen params: {frozen}, trainable: {trainable}, "
              f"projection trainable: {keep_projection_trainable})")

    def unfreeze_top_blocks(self, n: int = 2):
        """
        Unfreeze the top *n* transformer blocks + layer-norm + projection
        for gradual fine-tuning.
        """
        # Unfreeze the last n blocks
        num_blocks = len(self.blocks)
        for i in range(max(0, num_blocks - n), num_blocks):
            for p in self.blocks[i].parameters():
                p.requires_grad = True
        # Always unfreeze final norm + projection
        for p in self.norm.parameters():
            p.requires_grad = True
        for p in self.projection.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[✓] Unfroze top {n} transformer blocks + norm + projection  "
              f"({trainable:,} trainable params)")

    def unfreeze_all(self):
        """Unfreeze every parameter in the macro stream."""
        for p in self.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        print(f"[✓] Macro stream fully unfrozen ({total:,} params)")

    # ── Forward ──────────────────────────────────────────────────────────
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim embedding (no classification head)."""
        B = x.shape[0]
        x = self.patch_embed(x)                         # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                # CLS token
        return self.projection(cls_out)                   # (B, output_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Forward pass.

        In **feature-extractor mode** → always returns 256-dim embedding.
        In **pretraining mode**:
            • ``return_embedding=False`` → emotion logits  (B, 7)
            • ``return_embedding=True``  → (embedding, logits)
        """
        emb = self.forward_features(x)                   # (B, 256)

        if self._feature_extractor_mode:
            return emb                                    # embedding only

        logits = self.fer_head(emb)                       # (B, 7)
        if return_embedding:
            return emb, logits
        return logits
