"""
Truth in a Blink — Full Dual-Stream Model
===========================================
Wraps macro stream, micro stream, fusion module, and classifier
into a single ``nn.Module`` for convenient training / inference.
"""

import torch
import torch.nn as nn

from .macro_stream import MacroStreamViT
from .micro_stream import MicroStreamTransformer
from .fusion import AttentionFusion
from .classifier import DeceptionClassifier


class DualStreamDeceptionDetector(nn.Module):
    """
    End-to-end dual-stream deception detection model.

    Inputs
    ------
    face_image : (B, 3, 224, 224)  — cropped face for macro stream.
    flow_seq   : (B, T, 2, H, W)   — optical-flow sequence for micro stream.

    Outputs
    -------
    prob       : (B, 1) — deception probability.
    w_macro    : (B, 1) — interpretable macro weight.
    w_micro    : (B, 1) — interpretable micro weight.
    macro_emb  : (B, 256) — macro embedding (for analysis).
    micro_emb  : (B, 256) — micro embedding (for analysis).
    """

    def __init__(self, macro_cfg=None, micro_cfg=None,
                 fusion_cfg=None, classifier_cfg=None):
        super().__init__()

        # ── Macro stream ─────────────────────────────────────────────────
        mk = macro_cfg or {}
        self.macro_stream = MacroStreamViT(
            image_size=mk.get("image_size", 224),
            patch_size=mk.get("patch_size", 16),
            in_channels=mk.get("in_channels", 3),
            embed_dim=mk.get("embed_dim", 384),
            depth=mk.get("depth", 6),
            num_heads=mk.get("num_heads", 6),
            mlp_ratio=mk.get("mlp_ratio", 4.0),
            dropout=mk.get("dropout", 0.1),
            output_dim=mk.get("output_dim", 256),
            num_fer_classes=mk.get("num_fer_classes", 7),
        )

        # ── Micro stream ─────────────────────────────────────────────────
        mc = micro_cfg or {}
        self.micro_stream = MicroStreamTransformer(
            flow_channels=mc.get("flow_channels", 2),
            motion_descriptor_dim=mc.get("motion_descriptor_dim", 128),
            seq_len=mc.get("seq_len", 16),
            embed_dim=mc.get("embed_dim", 256),
            depth=mc.get("depth", 4),
            num_heads=mc.get("num_heads", 4),
            mlp_ratio=mc.get("mlp_ratio", 4.0),
            dropout=mc.get("dropout", 0.1),
            output_dim=mc.get("output_dim", 256),
        )

        # ── Fusion ────────────────────────────────────────────────────────
        fc = fusion_cfg or {}
        self.fusion = AttentionFusion(
            macro_dim=fc.get("macro_dim", 256),
            micro_dim=fc.get("micro_dim", 256),
            hidden_dim=fc.get("hidden_dim", 256),
            num_heads=fc.get("num_heads", 4),
            dropout=fc.get("dropout", 0.1),
        )

        # ── Classifier ───────────────────────────────────────────────────
        cc = classifier_cfg or {}
        self.classifier = DeceptionClassifier(
            input_dim=cc.get("input_dim", 256),
            hidden_dim=cc.get("hidden_dim", 128),
            dropout=cc.get("dropout", 0.3),
        )

    def forward(self, face_image: torch.Tensor, flow_seq: torch.Tensor):
        macro_emb = self.macro_stream.forward_features(face_image)  # (B, 256)
        micro_emb = self.micro_stream(flow_seq)                     # (B, 256)
        fused, w_macro, w_micro = self.fusion(macro_emb, micro_emb)
        prob = self.classifier(fused)                               # (B, 1)
        return prob, w_macro, w_micro, macro_emb, micro_emb

    def load_macro_pretrained(self, checkpoint_path: str):
        """
        Load FER2013-pretrained weights into the macro stream, then
        strip the emotion classification head and convert to
        feature-extractor mode.

        The ``fer_head`` weights are safely ignored so the backbone
        serves purely as an embedding network.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        # Load all weights (including fer_head) into the intact model
        self.macro_stream.load_state_dict(state, strict=False)
        print(f"[✓] Loaded FER2013-pretrained weights from {checkpoint_path}")

        # Convert to feature-extractor mode: remove fer_head
        self.macro_stream.to_feature_extractor()

    def freeze_macro(self, keep_projection_trainable: bool = True):
        """
        Freeze the macro backbone while keeping the projection layer
        trainable so it can adapt to the deception task.
        """
        self.macro_stream.freeze_backbone(
            keep_projection_trainable=keep_projection_trainable
        )

    def unfreeze_macro_top(self, n_blocks: int = 2):
        """
        Unfreeze the top *n* transformer blocks + projection for
        gradual fine-tuning (keeps early blocks frozen).
        """
        self.macro_stream.unfreeze_top_blocks(n_blocks)

    def unfreeze_macro(self):
        """Unfreeze every macro-stream parameter."""
        self.macro_stream.unfreeze_all()
