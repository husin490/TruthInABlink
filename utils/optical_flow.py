"""
Truth in a Blink — Enhanced Optical Flow Utilities
====================================================
Improved optical-flow computation with:
  • Normalisation: zero-mean, unit-variance, 99th-percentile outlier clipping
  • TV-L1 optical flow option (better edges, less smooth-field bias)
  • Farneback remains default (faster, no extra deps)
  • Drop-in compatible with existing compute_flow_sequence signature

Usage
-----
    # Direct replacement — import from here instead of micro_stream:
    from utils.optical_flow import compute_flow_sequence_normalised

    # Or CLI for batch pre-processing:
    python -m utils.optical_flow --clips-dir /path/to/Clips --output flows/
"""

import sys
import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Core optical-flow algorithms ─────────────────────────────────────────────

def compute_optical_flow_farneback(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """Dense Farneback optical-flow.  Returns (H, W, 2)."""
    return cv2.calcOpticalFlowFarneback(
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


def compute_optical_flow_tvl1(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """TV-L1 optical-flow (sharper motion boundaries).  Returns (H, W, 2)."""
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
        tau=0.25,
        lambda_=0.15,
        theta=0.3,
        nscales=5,
        warps=5,
        epsilon=0.01,
        innnerIterations=30,
        outerIterations=10,
        scaleStep=0.8,
        gamma=0.0,
        medianFiltering=5,
    )
    flow = tvl1.calc(prev_gray, curr_gray, None)
    return flow  # (H, W, 2)


def compute_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    method: Literal["farneback", "tvl1"] = "farneback",
) -> np.ndarray:
    """
    Dispatch to the chosen optical-flow method.

    Parameters
    ----------
    prev_gray, curr_gray : uint8 grayscale frames
    method : 'farneback' (default, fast) or 'tvl1' (sharper edges)

    Returns
    -------
    flow : (H, W, 2) float32
    """
    if method == "tvl1":
        return compute_optical_flow_tvl1(prev_gray, curr_gray)
    return compute_optical_flow_farneback(prev_gray, curr_gray)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_flow(
    flow: np.ndarray,
    percentile_clip: float = 99.0,
) -> np.ndarray:
    """
    Normalise optical-flow field: outlier clip → zero-mean → unit-variance.

    Parameters
    ----------
    flow : (H, W, 2) or (T, 2, H, W)
    percentile_clip : clip magnitudes above this percentile

    Returns
    -------
    normalised flow, same shape
    """
    if flow.size == 0:
        return flow

    # Work on a copy
    out = flow.astype(np.float32).copy()

    # 99th-percentile outlier clipping per channel
    for c in range(_flow_channels(out)):
        ch = _get_channel(out, c)
        if ch.size == 0:
            continue
        lo = np.percentile(ch, 100.0 - percentile_clip)
        hi = np.percentile(ch, percentile_clip)
        _set_channel(out, c, np.clip(ch, lo, hi))

    # Zero-mean, unit-variance (global)
    mu = out.mean()
    std = out.std()
    if std > 1e-8:
        out = (out - mu) / std

    return out


def _flow_channels(flow: np.ndarray) -> int:
    """Number of channels depending on layout."""
    if flow.ndim == 3 and flow.shape[-1] == 2:   # (H, W, 2)
        return 2
    if flow.ndim == 3 and flow.shape[0] == 2:    # (2, H, W)
        return 2
    if flow.ndim == 4 and flow.shape[1] == 2:    # (T, 2, H, W)
        return 2
    return 1


def _get_channel(flow: np.ndarray, c: int) -> np.ndarray:
    if flow.ndim == 3 and flow.shape[-1] == 2:
        return flow[..., c]
    if flow.ndim == 3 and flow.shape[0] == 2:
        return flow[c]
    if flow.ndim == 4 and flow.shape[1] == 2:
        return flow[:, c]
    return flow


def _set_channel(flow: np.ndarray, c: int, vals: np.ndarray):
    if flow.ndim == 3 and flow.shape[-1] == 2:
        flow[..., c] = vals
    elif flow.ndim == 3 and flow.shape[0] == 2:
        flow[c] = vals
    elif flow.ndim == 4 and flow.shape[1] == 2:
        flow[:, c] = vals


# ── Sequence computation ─────────────────────────────────────────────────────

def compute_flow_sequence_normalised(
    frames_gray: list[np.ndarray],
    target_size: tuple[int, int] = (56, 56),
    method: Literal["farneback", "tvl1"] = "farneback",
    normalise: bool = True,
    percentile_clip: float = 99.0,
) -> np.ndarray:
    """
    Compute normalised optical-flow between consecutive frames.

    Drop-in replacement for models.micro_stream.compute_flow_sequence with
    added normalisation and method selection.

    Parameters
    ----------
    frames_gray  : list of grayscale uint8 arrays
    target_size  : (H, W) to resize each flow field
    method       : 'farneback' or 'tvl1'
    normalise    : apply zero-mean/unit-var normalisation
    percentile_clip : outlier clipping percentile

    Returns
    -------
    flows : np.ndarray, shape (T-1, 2, H, W) — channel-first, float32
    """
    if len(frames_gray) < 2:
        return np.zeros((1, 2, *target_size), dtype=np.float32)

    flows = []
    for i in range(len(frames_gray) - 1):
        flow = compute_optical_flow(frames_gray[i], frames_gray[i + 1], method)
        flow_resized = cv2.resize(flow, (target_size[1], target_size[0]))
        flows.append(flow_resized.transpose(2, 0, 1))  # (2, H, W)

    stacked = np.stack(flows, axis=0).astype(np.float32)  # (T-1, 2, H, W)

    if normalise:
        stacked = normalise_flow(stacked, percentile_clip)

    return stacked


# ── Magnitude / angle visualisation helpers ───────────────────────────────────

def flow_to_rgb(flow_hw2: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 2) flow to an HSV-encoded RGB visualisation (H, W, 3) uint8.
    """
    mag, ang = cv2.cartToPolar(flow_hw2[..., 0], flow_hw2[..., 1])
    hsv = np.zeros((*flow_hw2.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flow_magnitude_stats(flow_seq: np.ndarray) -> dict:
    """
    Given (T, 2, H, W) flow, return temporal motion statistics.
    """
    # Magnitude per frame: sqrt(dx^2 + dy^2)
    mag = np.sqrt(flow_seq[:, 0] ** 2 + flow_seq[:, 1] ** 2)  # (T, H, W)
    mean_per_frame = mag.mean(axis=(1, 2))  # (T,)
    return {
        "mean_magnitude": float(mag.mean()),
        "std_magnitude": float(mag.std()),
        "max_magnitude": float(mag.max()),
        "temporal_mean": mean_per_frame.tolist(),
        "temporal_std": float(mean_per_frame.std()),
    }


# ── CLI: batch pre-compute normalised flows ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch pre-compute normalised optical flow")
    parser.add_argument("--clips-dir", type=str, required=True,
                        help="Path to Clips/ directory (with Deceptive/ and Truthful/)")
    parser.add_argument("--output", type=str, default="flows",
                        help="Output directory for .npy flow files")
    parser.add_argument("--method", choices=["farneback", "tvl1"], default="farneback")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--flow-size", type=int, nargs=2, default=[56, 56])
    parser.add_argument("--no-normalise", action="store_true")
    args = parser.parse_args()

    from data.rldd_dataset import sample_frames

    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(clips_dir.rglob("*.mp4"))
    print(f"Found {len(videos)} videos")

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        frames = sample_frames(cap, args.num_frames)
        cap.release()

        if len(frames) < 2:
            print(f"  [skip] {vid.name}: not enough frames")
            continue

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        flow_seq = compute_flow_sequence_normalised(
            grays,
            target_size=tuple(args.flow_size),
            method=args.method,
            normalise=not args.no_normalise,
        )

        # Save with relative structure preserved
        rel = vid.relative_to(clips_dir)
        save_path = out_dir / rel.with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, flow_seq)
        print(f"  [✓] {rel} → {flow_seq.shape}")

    print(f"\nDone. Flows saved in {out_dir}/")


if __name__ == "__main__":
    main()
