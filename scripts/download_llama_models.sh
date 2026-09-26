#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/.venv/bin/python" -m qwen3_tts_lab.artifacts \
  "$project_root/config/llama_model.json" "$project_root/models/qwen3-tts-llama"
