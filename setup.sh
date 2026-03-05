#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Truth in a Blink — macOS Setup Script (Apple Silicon)
# ──────────────────────────────────────────────────────────────────────────────
# Run this once to set up the Python environment and dependencies.
#
#   chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  Truth in a Blink — Environment Setup"
echo "  Optimised for Apple Silicon (M1/M2/M3)"
echo "═══════════════════════════════════════════════════════════"

# ── Check Python ─────────────────────────────────────────────────────────────
PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" &> /dev/null; then
    echo "❌ Python 3 not found. Install via: brew install python@3.11"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PY_VERSION"

# ── Create virtual environment ───────────────────────────────────────────────
VENV_DIR="$(dirname "$0")/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists."
fi

# Activate
source "$VENV_DIR/bin/activate"

# ── Upgrade pip ──────────────────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel

# ── Install PyTorch (MPS-enabled) ────────────────────────────────────────────
echo ""
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# ── Install remaining dependencies ───────────────────────────────────────────
echo ""
echo "Installing project dependencies..."
pip install -r "$(dirname "$0")/requirements.txt"

# ── Verify MPS availability ──────────────────────────────────────────────────
echo ""
echo "Verifying MPS (Metal) availability..."
$PYTHON -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  MPS available   : {torch.backends.mps.is_available()}')
print(f'  MPS built       : {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    x = torch.randn(2, 3, device='mps')
    print(f'  MPS tensor test : PASSED ({x.device})')
else:
    print('  ⚠ MPS not available — will fall back to CPU.')
"

# ── Create output directories ────────────────────────────────────────────────
mkdir -p "$(dirname "$0")/checkpoints"
mkdir -p "$(dirname "$0")/recordings"
mkdir -p "$(dirname "$0")/logs"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Activate environment:  source venv/bin/activate"
echo ""
echo "  Quick start:"
echo "    Stage 1: python -m training.train_fer"
echo "    Stage 2: python -m training.train_rldd"
echo "    Evaluate: python -m evaluation.evaluate --sweep"
echo "    Live UI:  streamlit run ui/dashboard.py"
echo "    CLI Demo: python -m inference.realtime_engine"
echo "═══════════════════════════════════════════════════════════"
