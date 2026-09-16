# Validated environment

Validated on 2026-09-16 for the Phase 1 SDPA baseline.

| Item | Measured value |
| --- | --- |
| Python | 3.12.14 (project-local `.venv`) |
| `qwen-tts` | 0.1.1 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| CUDA available | `True` |
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB |
| NVIDIA driver reported through WSL | 591.86 |
| Attention backend | SDPA (`flash_attn` absent) |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

## Installation decisions

The Ubuntu system Python was 3.10 and did not provide Python 3.12 through its
configured APT sources. `scripts/bootstrap.sh` therefore installs pinned
CPython 3.12.14 project-locally with uv 0.12.15 when no `python3.12` is
available. It does not change the system Python or install a Linux NVIDIA
driver.

The CUDA baseline was installed from the official PyTorch CUDA 12.1 index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.5.1+cu121' 'torchaudio==2.5.1+cu121'
```

No FlashAttention package was installed. The optional system audio dependency is
installed as `sox` 14.4.2 with `libsox-fmt-all` 14.4.2. The Python environment
was not changed for this addition.

## Phase 1 verification

- `nvidia-smi` detected the RTX 3060 with 12 GiB VRAM.
- `pytest -q` passed: 2 tests.
- `qwen_tts` imported successfully and `torch.cuda.is_available()` was true.
- The 1.7B Base model loaded with explicit SDPA on `cuda:0` without OOM.
  Peak CUDA memory was 4,203,216,384 bytes allocated and 4,334,813,184 bytes
  reserved.
- No `local_data/reference.wav` and `local_data/reference.txt` pair was present,
  so no voice-clone WAV was generated.

## Phase 1.5 verification

- `scripts/webui.sh` loaded the official Base-model Gradio UI with SDPA and
  `float16` on the cached model artifacts.
- The UI listened only on `127.0.0.1:8000` and returned HTTP 200. It was stopped
  after validation; no public sharing link or HTTPS endpoint was used.
