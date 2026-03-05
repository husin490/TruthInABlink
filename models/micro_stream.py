"""
Truth in a Blink — Micro Stream (Motion Analysis Transformer)
==============================================================
Extracts temporal motion cues from video via optical flow, builds
per-frame motion descriptors, and models them with a small transformer
encoder to produce a 256-dim embedding.

Pipeline
--------
1.  Compute dense optical-flow between consecutive frames (Farneback).
2.  Encode each flow field into a compact descriptor via a small CNN.
3.  Feed the sequence of descriptors into a transformer encoder.
4.  Pool → 256-dim embedding.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Optical-flow extraction (CPU, NumPy) ────────────────────────────────────

def compute_optical_flow(prev_gray: np.ndarray,
                         curr_gray: np.ndarray) -> np.ndarray:
    """
    Dense Farneback optical-flow.

    Returns
    -------
    flow : np.ndarray, shape (H, W, 2)  — dx, dy displacement.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow  # (H, W, 2)


def compute_flow_sequence(frames_gray: list[np.ndarray],
                          target_size: tuple[int, int] = (56, 56)
                          ) -> np.ndarray:
    """
    Compute optical-flow between consecutive frames and resize.

    Parameters
    ----------
    frames_gray : list of grayscale uint8 arrays.
    target_size : (H, W) to resize each flow field.

    Returns
    -------
    flows : np.ndarray, shape (T-1, 2, H, W) — channel-first flows.
    """
    flows = []
    for i in range(len(frames_gray) - 1):
        flow = compute_optical_flow(frames_gray[i], frames_gray[i + 1])
        flow_resized = cv2.resize(flow, (target_size[1], target_size[0]))
        flows.append(flow_resized.transpose(2, 0, 1))  # (2, H, W)
    return np.stack(flows, axis=0)  # (T-1, 2, H, W)


# ─── Motion Descriptor CNN ───────────────────────────────────────────────────

class MotionDescriptorCNN(nn.Module):
    """
    Lightweight CNN that reduces a single flow field (2, 56, 56)
    into a compact descriptor vector.
    """

    def __init__(self, in_channels: int = 2, descriptor_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),   # 28×28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # 14×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # 7×7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                               # 1×1
        )
        self.fc = nn.Linear(128, descriptor_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W) → (B, descriptor_dim)"""
        return self.fc(self.conv(x).flatten(1))


# ─── Temporal Transformer Encoder ────────────────────────────────────────────

class TemporalTransformerBlock(nn.Module):
    """Pre-norm transformer block using manual attention (MPS-compatible)."""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

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
        B, N, C = x.shape
        h = self.norm1(x)

        # Manual self-attention
        qkv = self.qkv(h).contiguous().reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()    # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        h = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        h = self.attn_proj(h)

        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MicroStreamTransformer(nn.Module):
    """
    Full micro-stream: CNN descriptor → temporal transformer → 256-dim embedding.

    Parameters
    ----------
    flow_channels         : Channels in optical-flow input (2).
    motion_descriptor_dim : CNN output dim per frame.
    seq_len               : Expected sequence length (for positional embedding).
    embed_dim             : Transformer hidden dimension.
    depth                 : Number of transformer blocks.
    num_heads             : Attention heads per block.
    mlp_ratio             : MLP expansion ratio.
    dropout               : Dropout probability.
    output_dim            : Final embedding dimension.
    """

    def __init__(
        self,
        flow_channels: int = 2,
        motion_descriptor_dim: int = 128,
        seq_len: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 256,
    ):
        super().__init__()
        self.descriptor_cnn = MotionDescriptorCNN(flow_channels,
                                                   motion_descriptor_dim)
        # Project descriptor to transformer dim
        self.input_proj = nn.Linear(motion_descriptor_dim, embed_dim)

        # Learnable [CLS] token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, flow_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        flow_seq : (B, T, 2, H, W) — sequence of optical-flow fields.

        Returns
        -------
        embedding : (B, output_dim)
        """
        B, T, C, H, W = flow_seq.shape

        # Encode each frame's flow with the CNN
        x = flow_seq.contiguous().reshape(B * T, C, H, W)  # (B*T, 2, H, W)
        x = self.descriptor_cnn(x)                           # (B*T, desc_dim)
        x = x.contiguous().reshape(B, T, -1)                 # (B, T, desc_dim)
        x = self.input_proj(x)                            # (B, T, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B, T+1, D)

        # Handle variable-length sequences with interpolated positional embed
        if x.size(1) != self.pos_embed.size(1):
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2).contiguous(),
                size=x.size(1),
                mode='linear',
                align_corners=False,
            ).transpose(1, 2).contiguous()
        else:
            pos = self.pos_embed

        x = self.pos_drop(x + pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                # CLS token
        return self.projection(cls_out)                   # (B, output_dim)
