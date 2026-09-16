#!/usr/bin/env bash
set -uo pipefail

failures=0

ok()   { printf '[OK]   %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

printf 'Qwen3-TTS environment doctor\n\n'

if grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null; then
  ok "Running under WSL"
else
  warn "WSL was not detected. This project targets WSL2."
fi

case "$PWD" in
  /mnt/*) warn "Repository is under $PWD. For ML workloads, the WSL Linux filesystem is usually preferable to /mnt/<drive>." ;;
  *) ok "Repository appears to be on the Linux filesystem: $PWD" ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi is available"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi query failed"
else
  fail "nvidia-smi is not available inside WSL. Check the Windows NVIDIA driver/WSL GPU integration; do not install a Linux display driver as a first fix."
fi

if command -v python3 >/dev/null 2>&1; then
  pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
  ok "python3: $pyver"
  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' ; then
    warn "Python 3.12 is the project target; bootstrap may need an explicit python3.12 interpreter."
  fi
else
  fail "python3 is missing"
fi

if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg: $(ffmpeg -version 2>/dev/null | head -n1)"
else
  warn "ffmpeg is not installed yet; bootstrap will install it."
fi

if command -v git >/dev/null 2>&1; then
  ok "git: $(git --version)"
else
  fail "git is missing"
fi

printf '\nMemory:\n'
free -h || true
printf '\nFilesystem:\n'
df -h . || true

if [ "$failures" -gt 0 ]; then
  printf '\nDoctor found %d blocking issue(s).\n' "$failures"
  exit 1
fi

printf '\nNo blocking pre-bootstrap issues detected.\n'
