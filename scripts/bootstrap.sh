#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "WARNING: WSL was not detected; this repository is designed for WSL2." >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not visible inside WSL." >&2
  echo "Fix Windows/WSL GPU integration first. Do not install a Linux NVIDIA display driver as the default remedy." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  pkg-config \
  git \
  ffmpeg \
  libsndfile1

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "ERROR: $PYTHON_BIN is not Python 3.12." >&2
  echo "Install/use Python 3.12, or run with PYTHON_BIN=/path/to/python3.12 ./scripts/bootstrap.sh" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'

python - <<'PY'
import sys
import torch
import qwen_tts

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit(
        "qwen-tts installed, but PyTorch cannot access CUDA. "
        "Install a current CUDA-enabled PyTorch build compatible with the WSL driver, using official PyTorch guidance."
    )
PY

echo
echo "Baseline environment installed. FlashAttention has intentionally NOT been installed yet."
echo "Next: source .venv/bin/activate && pytest -q"
