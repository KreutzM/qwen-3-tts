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

required_commands=(ffmpeg git)
missing_commands=()
for command in "${required_commands[@]}"; do
  command -v "$command" >/dev/null 2>&1 || missing_commands+=("$command")
done
if [ "${#missing_commands[@]}" -gt 0 ]; then
  echo "ERROR: missing system prerequisites: ${missing_commands[*]}" >&2
  echo "Install them with your distribution package manager, then rerun this script." >&2
  exit 1
fi

# Ubuntu 22.04 provides Python 3.10 by default.  Keep Python 3.12 project-local
# when the host does not already provide it, so bootstrap never needs an
# interactive sudo password or changes the WSL system Python.
UV_VERSION="${UV_VERSION:-0.12.15}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.14}"
TOOLS_DIR="$PWD/.tools"
UV_BIN="$TOOLS_DIR/uv"
PYTHON_DIR="$PWD/.python"

if [ -n "${PYTHON_BIN:-}" ]; then
  selected_python="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  selected_python="$(command -v python3.12)"
else
  if [ ! -x "$UV_BIN" ]; then
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
      echo "ERROR: Python 3.12 is unavailable and curl/tar are needed to install the local runtime." >&2
      exit 1
    fi
    mkdir -p "$TOOLS_DIR"
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "$temp_dir"' EXIT
    archive="$temp_dir/uv.tar.gz"
    curl -fsSL --retry 3 \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
      -o "$archive"
    tar -xzf "$archive" -C "$temp_dir"
    install -m 0755 "$temp_dir/uv-x86_64-unknown-linux-gnu/uv" "$UV_BIN"
  fi
  UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" "$UV_BIN" python install "$PYTHON_VERSION" >&2
  selected_python="$(UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" "$UV_BIN" python find "$PYTHON_VERSION")"
fi

if ! "$selected_python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "ERROR: $selected_python is not Python 3.12." >&2
  exit 1
fi

if [ -d .venv ] && ! .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
  echo "ERROR: existing .venv does not use Python 3.12." >&2
  echo "Move it aside or remove it explicitly, then rerun bootstrap." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$selected_python" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Use the official CUDA 12.1 wheel index for the baseline. It prevents pip
# from selecting a CPU-only PyTorch wheel; FlashAttention remains optional.
torch_install=(
  --index-url https://download.pytorch.org/whl/cu121
  'torch==2.5.1+cu121'
  'torchaudio==2.5.1+cu121'
)
if [ -n "${TORCH_WHEEL:-}" ]; then
  if [ ! -f "$TORCH_WHEEL" ]; then
    echo "ERROR: TORCH_WHEEL does not exist: $TORCH_WHEEL" >&2
    exit 1
  fi
  torch_install=(
    --index-url https://download.pytorch.org/whl/cu121
    "$TORCH_WHEEL"
    'torchaudio==2.5.1+cu121'
  )
fi
python -m pip install "${torch_install[@]}"
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
