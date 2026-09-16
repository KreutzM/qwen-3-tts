# Qwen3-TTS Local Lab

Local, reproducible workspace for evaluating **Qwen3-TTS** on German text and voice cloning under **Windows 11 + WSL2**.

## Target system

- CPU: AMD Ryzen 9 5950X
- RAM: 64 GB
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM
- Host: Windows 11
- Runtime: WSL2 / Ubuntu
- Primary model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Primary language: German

## Project goals

1. Establish a clean and reproducible WSL2/CUDA/Python environment.
2. Run Qwen3-TTS locally on the RTX 3060.
3. Evaluate German speech quality and voice cloning.
4. Measure speed, VRAM use and long-form stability.
5. Build a foundation for later audiobook / EPUB / DAISY experiments.

This repository contains **project code, setup automation, tests and experiment metadata only**. Model weights, generated audio and private/reference voice recordings stay local and are not committed.

## Start here

For humans: read [`AGENTS.md`](AGENTS.md), [`TASKS.md`](TASKS.md) and [`docs/SETUP_WSL2.md`](docs/SETUP_WSL2.md).

For Codex CLI: start in the repository root and instruct Codex to read `AGENTS.md` and execute **Phase 1** in `TASKS.md`. Codex should validate each step on the actual machine rather than assuming CUDA/PyTorch details.

## Intended workflow

```bash
./scripts/doctor.sh
./scripts/bootstrap.sh
source .venv/bin/activate
pytest -q
```

After a local reference voice is placed in `local_data/` (ignored by Git), a voice-clone smoke test can be run with the project CLI described in the setup guide.

## Interactive local UI

After the baseline is installed, start the official Base-model Gradio UI with:

```bash
./scripts/webui.sh
```

Open <http://localhost:8000> from Windows. The server binds only to WSL loopback,
uses the validated CUDA/SDPA path, and does not create a public share link or
HTTPS endpoint. See the setup guide for reference-audio handling.

## Upstream

Qwen3-TTS is developed by the Qwen team: <https://github.com/QwenLM/Qwen3-TTS>

The upstream project documents German as a supported language and provides the 1.7B Base model for rapid voice cloning. This repository does not fork or vendor upstream Qwen3-TTS; it consumes the official `qwen-tts` package/model artifacts.
