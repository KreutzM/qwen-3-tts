#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

venv_python=".venv/bin/python"
venv_demo=".venv/bin/qwen-tts-demo"
model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
bind_ip="127.0.0.1"
port="8000"

[ -x "$venv_python" ] || fail "Missing project virtual environment. Run ./scripts/bootstrap.sh first."
[ -x "$venv_demo" ] || fail "qwen-tts-demo is missing from .venv. Run ./scripts/bootstrap.sh first."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable. Fix WSL GPU integration; do not install a Linux NVIDIA display driver."
command -v sox >/dev/null 2>&1 || fail "sox is unavailable. Install it with: sudo apt-get install -y sox libsox-fmt-all"

"$venv_python" - <<'PY' || exit 1
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}. Run ./scripts/bootstrap.sh first.") from exc

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: PyTorch cannot access CUDA. Run ./scripts/doctor.sh and verify the validated CUDA-enabled PyTorch environment."
    )

print(f"CUDA ready: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__}, CUDA {torch.version.cuda})")
PY

if command -v ss >/dev/null 2>&1 && ss -ltnH "sport = :$port" | grep -q .; then
  fail "Port $port is already listening. Stop the existing service or choose a different port in scripts/webui.sh."
fi

printf 'Starting Qwen3-TTS UI at http://localhost:%s (WSL bind: %s)\n' "$port" "$bind_ip"
printf 'Press Ctrl-C to stop the server.\n'
exec "$venv_demo" "$model_id" \
  --device cuda:0 \
  --dtype float16 \
  --no-flash-attn \
  --ip "$bind_ip" \
  --port "$port" \
  --no-share
