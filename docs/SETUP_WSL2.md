# WSL2 setup guide

This is the intended baseline for the target workstation: Windows 11, Ryzen 9 5950X, 64 GB RAM and RTX 3060 12 GB.

## 1. Host/WSL prerequisites

Use a current NVIDIA Windows driver with WSL CUDA support. Inside WSL, `nvidia-smi` should show the RTX 3060.

Do **not** install a Linux NVIDIA display driver inside WSL as a routine setup step. WSL receives GPU access through the Windows driver.

Prefer cloning the repository into the Linux filesystem, for example:

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/KreutzM/qwen-3-tts.git
cd qwen-3-tts
```

## 2. Inspect the machine first

```bash
./scripts/doctor.sh
```

Resolve blocking GPU/WSL issues before installing the Python stack.

## 3. Install the baseline

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
pytest -q
```

The project intentionally starts without FlashAttention. A working standard PyTorch/SDPA path is easier to diagnose and becomes the comparison baseline.

If the `qwen-tts` dependency resolves to a PyTorch build without CUDA support, install a current CUDA-enabled PyTorch build compatible with the actual WSL/NVIDIA driver using official PyTorch instructions, then rerun the verification. Record the exact command in `docs/ENVIRONMENT.md`.

## 4. Optional local Web UI

After the environment works, upstream provides a local demo server for the Base model:

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 8000
```

Open `http://localhost:8000` from Windows. Keeping it bound to `127.0.0.1` avoids exposing the demo to the LAN unnecessarily.

## 5. Voice-clone smoke test

Create local directories (they are ignored by Git):

```bash
mkdir -p local_data outputs
```

Place a short, clean reference recording that you have the right to use at:

```text
local_data/reference.wav
```

Place its exact transcript at:

```text
local_data/reference.txt
```

Then run:

```bash
qwen3-clone \
  --ref-audio local_data/reference.wav \
  --ref-text-file local_data/reference.txt \
  --text-file experiments/texts/de_prose.txt \
  --out outputs/de_prose.wav
```

The first run may download several GB of model data into the normal model cache. Those files must not be committed.

## 6. FlashAttention — only after baseline success

Upstream recommends FlashAttention 2 to reduce GPU memory use. Treat it as a separate optimization experiment, because build compatibility depends on the current PyTorch/CUDA toolchain.

If the validated environment is compatible, the upstream installation pattern is:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Then rerun the same smoke/benchmark workload and compare VRAM use and real-time factor. Keep the SDPA fallback.

## 7. Reproducibility record

After a successful baseline, create `docs/ENVIRONMENT.md` with measured versions and generate a reviewed lock snapshot:

```bash
python -m pip freeze --exclude-editable > requirements.lock.txt
```

Also document any non-default PyTorch package index or install command because a freeze file may not preserve wheel origin.
