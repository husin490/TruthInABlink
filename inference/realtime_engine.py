"""
Truth in a Blink — Real-Time Inference Engine
===============================================
Webcam-based pipeline that:
  1. Captures frames and detects faces
  2. Maintains a motion buffer for optical-flow computation
  3. Runs the dual-stream model on the latest data
  4. Outputs smoothed deception probability + decision

Designed to run on Apple M3 Max with MPS acceleration.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
import collections
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, CHECKPOINT_DIR
from models.dual_stream import DualStreamDeceptionDetector
from models.micro_stream import compute_optical_flow
from utils.helpers import get_device, EMASmooth, classify_deception


# ─── Face Transform (pre-built for speed) ────────────────────────────────────

_face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Haar Cascade for Face Detection ─────────────────────────────────────────

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


class RealtimeInferenceEngine:
    """
    Manages the real-time inference loop.

    Attributes
    ----------
    model         : DualStreamDeceptionDetector on target device.
    device        : torch.device (MPS / CUDA / CPU).
    buffer_size   : Number of grayscale frames for the motion buffer.
    face_padding  : Fractional pad around detected face bounding box.
    smoother      : EMA smoother for probability output.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        buffer_size: int = 16,
        face_padding: float = 0.3,
        smoothing_alpha: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device(cfg.training.use_mps)
        self.buffer_size = buffer_size
        self.face_padding = face_padding
        self.smoother = EMASmooth(alpha=smoothing_alpha)

        # Motion buffer: stores grayscale frames
        self.gray_buffer: collections.deque = collections.deque(
            maxlen=buffer_size + 1   # need N+1 frames for N flow fields
        )

        # ── Load model ───────────────────────────────────────────────────
        self.model = DualStreamDeceptionDetector()
        self.model.macro_stream.to_feature_extractor()  # strip fer_head
        self.model = self.model.to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device,
                               weights_only=False)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            print(f"[✓] Loaded model from {checkpoint_path}")
        else:
            print("[!] No checkpoint loaded — using random weights.")

        # Latest results (thread-safe read by UI)
        self.last_prob: float = 0.5
        self.last_raw_prob: float = 0.5
        self.last_decision: str = "UNCERTAIN"
        self.last_w_macro: float = 0.5
        self.last_w_micro: float = 0.5
        self.last_face_crop: Optional[np.ndarray] = None
        self.fps: float = 0.0

    # ── Face detection ────────────────────────────────────────────────────

    def detect_face(self, frame_bgr: np.ndarray
                    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """
        Returns (face_crop_rgb, bbox) or (None, None).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        H, W = frame_bgr.shape[:2]
        pad_w, pad_h = int(w * self.face_padding), int(h * self.face_padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(W, x + w + pad_w)
        y2 = min(H, y + h + pad_h)

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb, (x1, y1, x2, y2)

    # ── Process one frame ─────────────────────────────────────────────────

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a single BGR frame from the webcam.

        Returns dict with:
            prob, raw_prob, decision, w_macro, w_micro, face_detected, bbox
        """
        t0 = time.time()

        # Face extraction
        face_crop, bbox = self.detect_face(frame_bgr)
        if face_crop is not None:
            self.last_face_crop = face_crop
            face_tensor = _face_transform(face_crop).unsqueeze(0).to(self.device)
        else:
            # Use last known face or zero tensor
            if self.last_face_crop is not None:
                face_tensor = _face_transform(self.last_face_crop).unsqueeze(0).to(self.device)
            else:
                face_tensor = torch.zeros(1, 3, 224, 224, device=self.device)

        # Update motion buffer
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (112, 112))
        self.gray_buffer.append(gray_small)

        # Build flow sequence
        if len(self.gray_buffer) >= 2:
            flows = []
            buffer_list = list(self.gray_buffer)
            for i in range(len(buffer_list) - 1):
                flow = compute_optical_flow(buffer_list[i], buffer_list[i + 1])
                flow_resized = cv2.resize(flow, (56, 56))
                flows.append(flow_resized.transpose(2, 0, 1))  # (2, 56, 56)
            flow_np = np.stack(flows, axis=0)                   # (T, 2, 56, 56)
            flow_tensor = torch.from_numpy(flow_np).float().unsqueeze(0).to(self.device)
        else:
            flow_tensor = torch.zeros(1, 1, 2, 56, 56, device=self.device)

        # ── Model inference ──────────────────────────────────────────────
        prob, w_macro, w_micro, _, _ = self.model(face_tensor, flow_tensor)

        raw_p = prob.item()
        smoothed_p = self.smoother.update(raw_p)

        high = cfg.decision.high_threshold
        low  = cfg.decision.low_threshold
        decision = classify_deception(smoothed_p, high, low)

        # Store latest
        self.last_prob = smoothed_p
        self.last_raw_prob = raw_p
        self.last_decision = decision
        self.last_w_macro = w_macro.item()
        self.last_w_micro = w_micro.item()

        elapsed = time.time() - t0
        self.fps = 1.0 / max(elapsed, 1e-6)

        return {
            "prob": smoothed_p,
            "raw_prob": raw_p,
            "decision": decision,
            "w_macro": self.last_w_macro,
            "w_micro": self.last_w_micro,
            "face_detected": face_crop is not None,
            "bbox": bbox,
            "fps": self.fps,
        }

    def reset(self):
        """Clear buffers and smoother state."""
        self.gray_buffer.clear()
        self.smoother.reset()
        self.last_prob = 0.5
        self.last_raw_prob = 0.5
        self.last_decision = "UNCERTAIN"
        self.last_face_crop = None


# ─── Standalone CLI demo ─────────────────────────────────────────────────────

def run_cli_demo(checkpoint_path: Optional[str] = None):
    """Simple OpenCV window demo (no Streamlit)."""
    engine = RealtimeInferenceEngine(
        checkpoint_path=checkpoint_path,
        buffer_size=cfg.inference.motion_buffer_size,
        face_padding=cfg.inference.face_padding,
        smoothing_alpha=cfg.inference.smoothing_alpha,
    )

    cap = cv2.VideoCapture(cfg.inference.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[Camera] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = engine.process_frame(frame)

        # Draw overlay
        color_map = {
            "DECEPTIVE": (0, 0, 255),
            "TRUTHFUL": (0, 200, 0),
            "UNCERTAIN": (0, 180, 255),
        }
        color = color_map.get(result["decision"], (255, 255, 255))

        # Draw bounding box
        if result["bbox"]:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Overlay text
        cv2.putText(frame,
                    f"{result['decision']}  P={result['prob']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame,
                    f"Macro={result['w_macro']:.2f}  Micro={result['w_micro']:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"FPS: {result['fps']:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Truth in a Blink — Real-Time Lie Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    args = parser.parse_args()
    run_cli_demo(args.checkpoint)
