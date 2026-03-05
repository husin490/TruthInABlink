"""
Truth in a Blink — Professional Streamlit Dashboard
=====================================================
A comprehensive real-time deception detection UI with:
  • Live webcam feed with face detection overlay
  • Real-time probability gauge & colour-coded status
  • Micro vs Macro contribution display
  • Probability-over-time chart & fusion weight trends
  • Recording controls (video, snapshot, session log)
  • Evaluation tab with confusion matrix & metrics
  • Settings panel for all adjustable parameters

Launch
------
    streamlit run ui/dashboard.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import json
import csv
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch
import streamlit as st
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, CHECKPOINT_DIR, RECORDING_DIR, LOG_DIR, RLDD_CLIPS
from inference.realtime_engine import RealtimeInferenceEngine
from utils.helpers import classify_deception

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=cfg.ui.page_title,
    page_icon=cfg.ui.page_icon,
    layout=cfg.ui.layout,
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for professional look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .status-deceptive {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        color: white; padding: 12px 24px; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    .status-truthful {
        background: linear-gradient(135deg, #2ed573, #7bed9f);
        color: white; padding: 12px 24px; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    .status-uncertain {
        background: linear-gradient(135deg, #ffa502, #ffbe76);
        color: white; padding: 12px 24px; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 15px rgba(255,165,2,0.4);
    }
    .metric-card {
        background: #1e1e2e; border-radius: 12px; padding: 16px;
        border: 1px solid #333; text-align: center;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem; color: #999; margin-top: 4px;
    }
    .gauge-container {
        text-align: center; padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "engine": None,
        "running": False,
        "prob_history": deque(maxlen=300),
        "w_macro_history": deque(maxlen=300),
        "w_micro_history": deque(maxlen=300),
        "time_history": deque(maxlen=300),
        "session_log": [],
        "recording": False,
        "video_writer": None,
        "record_start": None,
        "frame_count": 0,
        # Settings
        "high_threshold": cfg.decision.high_threshold,
        "low_threshold": cfg.decision.low_threshold,
        "smoothing_alpha": cfg.inference.smoothing_alpha,
        "buffer_size": cfg.inference.motion_buffer_size,
        "face_padding": cfg.inference.face_padding,
        "camera_index": cfg.inference.camera_index,
        # High Precision Mode
        "high_precision_mode": False,
        "_hp_saved_high": cfg.decision.high_threshold,
        "_hp_saved_low": cfg.decision.low_threshold,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create / get inference engine
# ─────────────────────────────────────────────────────────────────────────────

def get_engine() -> RealtimeInferenceEngine:
    if st.session_state.engine is None:
        ckpt = CHECKPOINT_DIR / "dual_stream_best.pt"
        ckpt_str = str(ckpt) if ckpt.exists() else None
        st.session_state.engine = RealtimeInferenceEngine(
            checkpoint_path=ckpt_str,
            buffer_size=st.session_state.buffer_size,
            face_padding=st.session_state.face_padding,
            smoothing_alpha=st.session_state.smoothing_alpha,
        )
    return st.session_state.engine


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Settings Panel
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### Decision Thresholds")

    # ── High Precision Mode toggle ───────────────────────────────────
    hp_mode = st.toggle(
        "🎯 High Precision Mode",
        value=st.session_state.high_precision_mode,
        help="Raises thresholds to ≥ 0.80/≤ 0.20 for fewer false positives. "
             "Increases UNCERTAIN zone — only strong signals produce verdicts.",
    )
    if hp_mode != st.session_state.high_precision_mode:
        st.session_state.high_precision_mode = hp_mode
        if hp_mode:
            # Save current thresholds and switch to high-precision
            st.session_state._hp_saved_high = st.session_state.high_threshold
            st.session_state._hp_saved_low = st.session_state.low_threshold
            st.session_state.high_threshold = max(0.80, st.session_state.high_threshold)
            st.session_state.low_threshold = min(0.20, st.session_state.low_threshold)
        else:
            # Restore previous thresholds
            st.session_state.high_threshold = st.session_state._hp_saved_high
            st.session_state.low_threshold = st.session_state._hp_saved_low

    if st.session_state.high_precision_mode:
        st.info("🎯 High Precision Mode active — wider UNCERTAIN zone")

    st.session_state.high_threshold = st.slider(
        "High threshold (→ DECEPTIVE)", 0.50, 0.95,
        st.session_state.high_threshold, 0.05,
    )
    st.session_state.low_threshold = st.slider(
        "Low threshold (→ TRUTHFUL)", 0.05, 0.50,
        st.session_state.low_threshold, 0.05,
    )

    st.markdown("### Signal Processing")
    st.session_state.smoothing_alpha = st.slider(
        "Smoothing strength (EMA α)", 0.05, 1.0,
        st.session_state.smoothing_alpha, 0.05,
        help="Lower = smoother, Higher = more responsive",
    )
    st.session_state.buffer_size = st.slider(
        "Motion buffer size", 4, 32,
        st.session_state.buffer_size, 2,
    )

    st.markdown("### Face Detection")
    st.session_state.face_padding = st.slider(
        "Face crop padding", 0.0, 0.8,
        st.session_state.face_padding, 0.05,
    )
    st.markdown("### Camera Source")
    # Detect available cameras
    _available_cameras = []
    for _ci in range(11):
        _cap = cv2.VideoCapture(_ci)
        if _cap.isOpened():
            _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _available_cameras.append((_ci, f"Camera {_ci}  ({_w}×{_h})"))
            _cap.release()
        else:
            _cap.release()
    if not _available_cameras:
        _available_cameras = [(0, "Camera 0 (default)")]

    _cam_options = {label: idx for idx, label in _available_cameras}
    _current_label = next(
        (label for idx, label in _available_cameras
         if idx == st.session_state.camera_index),
        _available_cameras[0][1],
    )
    selected_camera = st.selectbox(
        "Select camera",
        list(_cam_options.keys()),
        index=list(_cam_options.keys()).index(_current_label)
            if _current_label in _cam_options else 0,
        help="Choose built-in, external, or iPhone (Continuity) camera",
    )
    st.session_state.camera_index = _cam_options[selected_camera]

    st.markdown("### Model")
    available_ckpts = list(CHECKPOINT_DIR.glob("*.pt"))
    if available_ckpts:
        selected = st.selectbox(
            "Checkpoint",
            [p.name for p in available_ckpts],
            index=0,
        )
    else:
        st.info("No checkpoints found. Train the model first.")

    st.markdown("---")
    st.markdown(
        "**Truth in a Blink** v2.0 (CP2+)  \n"
        "Visual Sensor-Based  \n"
        "Lie Detection"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Tabs
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🔍 Truth in a Blink</p>', unsafe_allow_html=True)
st.caption("Visual Sensor-Based Lie Detection Through Facial Expression Analysis")

tab_live, tab_viz, tab_record, tab_eval, tab_about = st.tabs([
    "🎥 Live Detection", "📊 Visualisation", "🎬 Recording",
    "📈 Evaluation", "ℹ️ About",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Detection
# ═════════════════════════════════════════════════════════════════════════════

with tab_live:
    col_ctrl, col_status = st.columns([2, 1])

    with col_ctrl:
        c1, c2, c3 = st.columns(3)
        start_btn = c1.button("▶ Start", use_container_width=True, type="primary")
        stop_btn  = c2.button("⏹ Stop", use_container_width=True)
        reset_btn = c3.button("🔄 Reset", use_container_width=True)

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    if reset_btn:
        st.session_state.running = False
        st.session_state.prob_history.clear()
        st.session_state.w_macro_history.clear()
        st.session_state.w_micro_history.clear()
        st.session_state.time_history.clear()
        if st.session_state.engine:
            st.session_state.engine.reset()

    # ── Live display area ────────────────────────────────────────────────
    col_video, col_panel = st.columns([3, 2])

    with col_video:
        video_placeholder = st.empty()
        fps_placeholder = st.empty()

    with col_panel:
        status_placeholder = st.empty()
        prob_placeholder = st.empty()
        gauge_placeholder = st.empty()

        st.markdown("#### Stream Contributions")
        weights_placeholder = st.empty()

    # ── Main loop ────────────────────────────────────────────────────────
    if st.session_state.running:
        engine = get_engine()
        engine.smoother.alpha = st.session_state.smoothing_alpha
        cfg.decision.high_threshold = st.session_state.high_threshold
        cfg.decision.low_threshold = st.session_state.low_threshold

        cap = cv2.VideoCapture(st.session_state.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("❌ Cannot open camera. Check camera index in Settings.")
        else:
            frame_counter = 0
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera frame lost.")
                    break

                result = engine.process_frame(frame)
                frame_counter += 1

                # Recording
                if st.session_state.recording and st.session_state.video_writer:
                    st.session_state.video_writer.write(frame)

                # ── Draw overlay on frame ────────────────────────────────
                display = frame.copy()
                color_map = {
                    "DECEPTIVE": (0, 0, 255),
                    "TRUTHFUL": (0, 200, 0),
                    "UNCERTAIN": (0, 180, 255),
                }
                color = color_map.get(result["decision"], (255, 255, 255))

                if result["bbox"]:
                    x1, y1, x2, y2 = result["bbox"]
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, result["decision"],
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)

                # Convert BGR → RGB for Streamlit
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_rgb, channels="RGB",
                                        use_container_width=True)

                fps_placeholder.caption(
                    f"FPS: {result['fps']:.1f} | "
                    f"Buffer: {len(engine.gray_buffer)}/{engine.buffer_size + 1} | "
                    f"Face: {'✓' if result['face_detected'] else '✗'}"
                )

                # ── Status panel ─────────────────────────────────────────
                css_class = f"status-{result['decision'].lower()}"
                status_placeholder.markdown(
                    f'<div class="{css_class}">{result["decision"]}</div>',
                    unsafe_allow_html=True,
                )

                # Probability display
                prob_placeholder.markdown(
                    f"**Deception Probability:** `{result['prob']:.3f}`  \n"
                    f"Raw: `{result['raw_prob']:.3f}`"
                )

                # ── UNCERTAIN reason panel ───────────────────────────────
                if result["decision"] == "UNCERTAIN":
                    hi_t = st.session_state.high_threshold
                    lo_t = st.session_state.low_threshold
                    p = result["prob"]
                    gap_high = hi_t - p
                    gap_low = p - lo_t
                    closer_to = "DECEPTIVE" if gap_high < gap_low else "TRUTHFUL"
                    reason_parts = []
                    reason_parts.append(
                        f"Probability {p:.3f} falls in the uncertain zone "
                        f"({lo_t:.2f} – {hi_t:.2f})."
                    )
                    reason_parts.append(f"Leans **{closer_to}** (gap: {min(gap_high, gap_low):.3f}).")
                    if not result.get("face_detected", True):
                        reason_parts.append("⚠ No face detected — signal unreliable.")
                    if len(engine.gray_buffer) < engine.buffer_size:
                        reason_parts.append(
                            f"Motion buffer filling ({len(engine.gray_buffer)}/{engine.buffer_size})."
                        )
                    if st.session_state.high_precision_mode:
                        reason_parts.append("🎯 High Precision Mode — wider UNCERTAIN zone.")
                    gauge_placeholder.warning("  \n".join(reason_parts))
                else:
                    # Gauge (progress bar) for non-UNCERTAIN
                    gauge_placeholder.progress(
                        min(max(result["prob"], 0.0), 1.0),
                        text=f"P(Deception) = {result['prob']:.1%}",
                    )

                # Weights bars
                weights_placeholder.markdown(
                    f"**Macro (face):** {result['w_macro']:.1%}  \n"
                    f"**Micro (motion):** {result['w_micro']:.1%}"
                )

                # ── History ──────────────────────────────────────────────
                now = time.time()
                st.session_state.prob_history.append(result["prob"])
                st.session_state.w_macro_history.append(result["w_macro"])
                st.session_state.w_micro_history.append(result["w_micro"])
                st.session_state.time_history.append(now)

                # Session log entry (every 10 frames)
                if frame_counter % 10 == 0:
                    st.session_state.session_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "prob": round(result["prob"], 4),
                        "decision": result["decision"],
                        "w_macro": round(result["w_macro"], 4),
                        "w_micro": round(result["w_micro"], 4),
                        "fps": round(result["fps"], 1),
                    })

                # Small sleep to allow Streamlit to refresh
                time.sleep(0.03)

            cap.release()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Visualisation
# ═════════════════════════════════════════════════════════════════════════════

with tab_viz:
    st.markdown("### 📊 Real-Time Visualisation")

    if len(st.session_state.prob_history) > 0:
        # ── Probability over time ────────────────────────────────────────
        st.markdown("#### Deception Probability Over Time")
        prob_df = pd.DataFrame({
            "Probability": list(st.session_state.prob_history),
            "High Threshold": [st.session_state.high_threshold] * len(st.session_state.prob_history),
            "Low Threshold": [st.session_state.low_threshold] * len(st.session_state.prob_history),
        })
        st.line_chart(prob_df, height=300)

        # ── Fusion weight trends ─────────────────────────────────────────
        st.markdown("#### Fusion Weight Trends")
        weight_df = pd.DataFrame({
            "Macro (Face)": list(st.session_state.w_macro_history),
            "Micro (Motion)": list(st.session_state.w_micro_history),
        })
        st.area_chart(weight_df, height=250)

        # ── Summary statistics ───────────────────────────────────────────
        st.markdown("#### Session Statistics")
        probs = list(st.session_state.prob_history)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Prob", f"{np.mean(probs):.3f}")
        c2.metric("Std Dev", f"{np.std(probs):.3f}")
        c3.metric("Max Prob", f"{np.max(probs):.3f}")
        c4.metric("Min Prob", f"{np.min(probs):.3f}")

        # Decision distribution
        decisions = [classify_deception(p, st.session_state.high_threshold,
                                        st.session_state.low_threshold)
                     for p in probs]
        dist = {d: decisions.count(d) for d in ["TRUTHFUL", "UNCERTAIN", "DECEPTIVE"]}
        st.markdown("#### Decision Distribution")
        dist_df = pd.DataFrame({"Decision": list(dist.keys()),
                                "Count": list(dist.values())})
        st.bar_chart(dist_df.set_index("Decision"), height=200)
    else:
        st.info("Start the live detection to see visualisations here.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Recording & Dataset Tools
# ═════════════════════════════════════════════════════════════════════════════

with tab_record:
    st.markdown("### 🎬 Recording & Dataset Tools")

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.markdown("#### Video Recording")
        rec_label = st.selectbox("Label for this recording",
                                 ["truthful", "deceptive"])

        c1, c2 = st.columns(2)
        if c1.button("🔴 Start Recording", use_container_width=True):
            RECORDING_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{rec_label}_{timestamp}.mp4"
            filepath = RECORDING_DIR / filename

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(filepath), fourcc, 20.0, (640, 480))
            st.session_state.video_writer = writer
            st.session_state.recording = True
            st.session_state.record_start = datetime.now()
            st.success(f"Recording started → {filename}")

        if c2.button("⏹ Stop Recording", use_container_width=True):
            if st.session_state.video_writer:
                st.session_state.video_writer.release()
                st.session_state.video_writer = None
            st.session_state.recording = False
            st.success("Recording stopped and saved.")

            # Append to dataset CSV
            csv_path = RECORDING_DIR / "recordings_log.csv"
            file_exists = csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["filename", "label", "timestamp",
                                     "mean_prob", "duration_s"])
                probs = list(st.session_state.prob_history)
                mean_p = np.mean(probs) if probs else 0.0
                dur = 0
                if st.session_state.record_start:
                    dur = (datetime.now() - st.session_state.record_start).total_seconds()
                writer.writerow([
                    f"{rec_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    rec_label,
                    datetime.now().isoformat(),
                    f"{mean_p:.4f}",
                    f"{dur:.1f}",
                ])

        if st.session_state.recording:
            st.warning("🔴 Recording in progress…")

    with col_rec2:
        st.markdown("#### Snapshot & Logs")

        if st.button("📸 Capture Snapshot", use_container_width=True):
            if st.session_state.engine and st.session_state.engine.last_face_crop is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = RECORDING_DIR / f"snapshot_{timestamp}.jpg"
                face_bgr = cv2.cvtColor(st.session_state.engine.last_face_crop,
                                        cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(snap_path), face_bgr)
                st.success(f"Snapshot saved → {snap_path.name}")
            else:
                st.warning("No face available. Start detection first.")

        if st.button("📋 Export Session Log", use_container_width=True):
            if st.session_state.session_log:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = LOG_DIR / f"session_log_{timestamp}.json"
                with open(log_path, "w") as f:
                    json.dump(st.session_state.session_log, f, indent=2)
                st.success(f"Session log exported → {log_path.name}")

                # Also offer download
                st.download_button(
                    "⬇️ Download Log",
                    data=json.dumps(st.session_state.session_log, indent=2),
                    file_name=f"session_log_{timestamp}.json",
                    mime="application/json",
                )
            else:
                st.info("No session data yet.")

    # Existing recordings
    st.markdown("---")
    st.markdown("#### Saved Recordings")
    recordings = list(RECORDING_DIR.glob("*.mp4"))
    if recordings:
        for r in recordings[-10:]:
            st.text(f"  📹 {r.name}  ({r.stat().st_size / 1_000_000:.1f} MB)")
    else:
        st.info("No recordings yet.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation
# ═════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown("### 📈 Model Evaluation on RLDD")

    eval_threshold = st.slider("Evaluation threshold", 0.1, 0.9, 0.5, 0.05)
    run_eval = st.button("🚀 Run Evaluation", type="primary")

    if run_eval:
        with st.spinner("Running evaluation…this may take a few minutes."):
            try:
                from evaluation.evaluate import evaluate_model, threshold_sweep
                from data.rldd_dataset import RLDDDataset
                from models.dual_stream import DualStreamDeceptionDetector
                from utils.helpers import get_device

                device = get_device(cfg.training.use_mps)
                model = DualStreamDeceptionDetector()
                model.macro_stream.to_feature_extractor()
                model = model.to(device)

                ckpt = CHECKPOINT_DIR / "dual_stream_best.pt"
                if ckpt.exists():
                    state = torch.load(ckpt, map_location=device, weights_only=False)
                    model.load_state_dict(state["model_state_dict"])
                    st.success("Model loaded from checkpoint.")
                else:
                    st.warning("No checkpoint found. Using random weights.")

                dataset = RLDDDataset(RLDD_CLIPS, num_frames=cfg.training.rldd_clip_frames)
                results = evaluate_model(model, dataset, device, eval_threshold)

                # ── Display metrics ──────────────────────────────────────
                st.markdown("#### Performance Metrics")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{results['accuracy']:.3f}")
                c2.metric("Precision", f"{results['precision']:.3f}")
                c3.metric("Recall", f"{results['recall']:.3f}")
                c4.metric("F1 Score", f"{results['f1']:.3f}")
                c5.metric("AUC", f"{results['auc']:.3f}")

                # Confusion matrix
                st.markdown("#### Confusion Matrix")
                cm = results["confusion_matrix"]
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Truthful", "Actual Deceptive"],
                    columns=["Predicted Truthful", "Predicted Deceptive"],
                )
                st.dataframe(cm_df, use_container_width=True)

                # Classification report
                st.markdown("#### Classification Report")
                st.code(results["classification_report"])

                # Fusion weights
                st.markdown("#### Average Fusion Weights")
                wc1, wc2 = st.columns(2)
                wc1.metric("Macro (Face)", f"{results['mean_w_macro']:.3f}")
                wc2.metric("Micro (Motion)", f"{results['mean_w_micro']:.3f}")

                # Per-clip results
                st.markdown("#### Per-Clip Results")
                clip_df = pd.DataFrame(results["per_clip"])
                st.dataframe(clip_df, use_container_width=True, height=300)

                # Threshold sweep
                st.markdown("#### Threshold Comparison")
                sweep = threshold_sweep(model, dataset, device)
                sweep_df = pd.DataFrame(sweep)
                st.dataframe(sweep_df, use_container_width=True)
                st.line_chart(sweep_df.set_index("threshold")[["accuracy", "f1"]], 
                              height=250)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ═════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.markdown("""
    ## About

    ### Truth in a Blink
    **Visual Sensor-Based Lie Detection Through Facial Expression Analysis**

    ---

    #### Architecture Overview

    | Component | Description |
    |-----------|-------------|
    | **Macro Stream** | Vision Transformer (ViT-Small) pretrained on FER2013 for facial context |
    | **Micro Stream** | Optical-flow CNN + Temporal Transformer for motion analysis |
    | **Fusion** | Cross-attention fusion with interpretable contribution weights |
    | **Classifier** | Binary MLP head outputting deception probability |

    #### Decision Strategy

    Three-state conservative output:
    - **DECEPTIVE** — probability ≥ high threshold
    - **TRUTHFUL** — probability ≤ low threshold
    - **UNCERTAIN** — in between

    #### Datasets

    1. **FER2013** — 35,887 facial expression images (7 emotions)
       - Used for macro stream pretraining
    2. **Real-Life Deception Detection 2016** — 121 video clips
       - 61 deceptive + 60 truthful
       - Used for dual-stream training and evaluation

    #### System Requirements

    - **Hardware:** Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM
    - **Acceleration:** PyTorch MPS (Metal Performance Shaders)
    - **Runs entirely offline** — no internet connection required

    #### Training Pipeline

    | Stage | Task | Dataset |
    |-------|------|---------|
    | 1 | Pretrain macro stream (emotion recognition) | FER2013 |
    | 2 | Train dual-stream (deception detection) | RLDD 2016 |
    | 3 | Optional fine-tuning on user-collected data | Custom |

    ---

    *Capstone Project Phase 2+ — Conference-Ready Implementation*

    #### CP2+ Enhancements

    | Feature | Description |
    |---------|-------------|
    | **Subject-wise K-Fold CV** | 5-fold splits ensuring no subject leakage between train/val/test |
    | **High Precision Mode** | Toggle in sidebar raises thresholds (≥0.80/≤0.20) for fewer false positives |
    | **UNCERTAIN Reason Panel** | When verdict is UNCERTAIN, explains *why* with distance to thresholds |
    | **MediaPipe Face Detection** | 3-tier fallback: MediaPipe → Haar → centre-crop |
    | **Normalised Optical Flow** | Zero-mean/unit-var with 99th-percentile outlier clipping |
    | **Temperature Calibration** | Post-hoc calibration for well-calibrated probability outputs |
    | **Ablation Study** | Quantifies contribution of each architectural component |
    """)
