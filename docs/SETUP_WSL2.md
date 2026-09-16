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

`bootstrap.sh` uses a system `python3.12` when one is available. On distributions
such as Ubuntu 22.04 that only ship an older system Python, it installs a pinned,
project-local CPython runtime under `.python/` using `uv`; both the runtime and
bootstrap helper are ignored by Git. This avoids changing the WSL system Python
or requiring an interactive `sudo` password. `ffmpeg` and `git` remain system
prerequisites.

The validated baseline uses the official PyTorch CUDA 12.1 index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.5.1+cu121' 'torchaudio==2.5.1+cu121'
```

The Windows driver reported through WSL must support CUDA 12.1. Do not replace
the Windows driver with a Linux driver inside WSL. Record the exact command in
`docs/ENVIRONMENT.md`.

## 4. Optional local Web UI

Install the optional SoX formats once if they are not already present:

```bash
sudo apt-get update
sudo apt-get install -y sox libsox-fmt-all
```

After the environment works, start the official Base-model Gradio UI through the
project wrapper:

```bash
./scripts/webui.sh
```

Open `http://localhost:8000` from Windows. The wrapper validates the project
environment and CUDA first, binds to `127.0.0.1`, uses SDPA (`--no-flash-attn`),
and disables Gradio sharing. HTTPS is not required for this localhost-only UI.

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

Do not add those files to Git. The transcript must match the recording exactly,
and the recording must be user-owned or used with consent.

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
