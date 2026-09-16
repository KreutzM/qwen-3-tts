# AGENTS.md

## Mission

Build and maintain a reproducible local Qwen3-TTS environment for German TTS and voice-cloning experiments on the target workstation. Prefer a small, inspectable project over a large framework.

## Target workstation

- Host: Windows 11
- Runtime: WSL2 / Ubuntu
- CPU: AMD Ryzen 9 5950X
- RAM: 64 GB
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM
- Primary model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Primary language: German

Do not assume that the installed WSL distribution, NVIDIA driver, PyTorch build, CUDA runtime or package versions match documentation. Detect and verify them on the actual machine.

## Sources of truth

1. This file and `TASKS.md` define local project intent and constraints.
2. The official upstream repository `QwenLM/Qwen3-TTS` is the source of truth for Qwen3-TTS APIs and supported models.
3. Official PyTorch/NVIDIA documentation takes precedence for current CUDA compatibility.
4. Record any machine-specific decisions in `docs/ENVIRONMENT.md` or `docs/DECISIONS.md` rather than relying on terminal history.

## Working rules

- Work inside the WSL Linux filesystem (for example `~/src/qwen-3-tts`), not under `/mnt/c`, unless there is a specific reason.
- Never install a Linux NVIDIA display driver inside WSL. GPU access should come from the Windows NVIDIA driver exposed to WSL.
- Start with the official `qwen-tts` Python package. Do not switch to community forks unless a concrete blocker is reproduced and documented.
- Establish a working baseline using PyTorch's standard attention path before adding FlashAttention. FlashAttention is an optimization, not a prerequisite.
- Prefer Python 3.12 and the project-local `.venv`.
- Make setup scripts idempotent. Re-running them should not destroy a working environment or user data.
- Do not commit model weights, Hugging Face caches, generated audio, private voice samples, access tokens, `.env` files or machine secrets.
- Reference voice audio must remain in `local_data/` and is ignored by Git. Use only audio the user has the right/consent to use.
- Do not fabricate a voice sample when none is present. A functional model-load/demo test can precede the real cloning test.
- Keep large downloads out of unit tests and GitHub Actions. GPU/model integration tests are local/manual unless explicitly redesigned later.
- Before changing dependency versions to fix a problem, capture the actual error and current versions.
- Avoid unnecessary refactors while establishing the baseline.
- Code and durable repo documentation should be in English. Report interactively to the user in German unless asked otherwise.

## Required workflow

Before making environment changes:

```bash
./scripts/doctor.sh
```

Then follow `TASKS.md`. For the initial baseline:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
pytest -q
```

After the environment works, record reproducibility data:

- Python version
- `qwen-tts` version
- PyTorch version
- CUDA availability and `torch.version.cuda`
- GPU name and VRAM
- NVIDIA driver version reported through WSL
- attention backend used
- exact Qwen model ID

Do not record hostnames, usernames, tokens or other unnecessary identifiers.

## Definition of done for Phase 1

Phase 1 is complete only when all applicable items are demonstrated on the target machine:

1. WSL2 sees the RTX 3060 via `nvidia-smi`.
2. A Python 3.12 `.venv` exists and the project installs successfully.
3. `torch.cuda.is_available()` is true.
4. `qwen_tts` imports successfully.
5. Unit tests pass.
6. The 1.7B Base model can be loaded on the GPU without out-of-memory failure.
7. If a user-owned reference sample exists, a short German voice-clone WAV is generated successfully.
8. Actual working versions and any deviations from the initial plan are documented.

## Optimization policy

Only after the baseline works:

- Try FlashAttention 2 if it is compatible with the installed stack.
- Measure whether it improves VRAM use and/or real-time factor on this RTX 3060.
- Keep an SDPA fallback.
- Never trade reproducibility for a small unmeasured optimization.

## Change quality

For code changes:

- Keep functions small and type-annotated where useful.
- Fail with actionable error messages.
- Use `pathlib.Path` for local paths.
- Never silently overwrite a user-provided reference recording.
- Generated files belong under `outputs/`.
- Add or update tests for logic that does not require downloading a model.
- Update `README.md`, `TASKS.md` or docs when commands or assumptions change.

Before concluding a task, run the relevant checks and summarize what was actually verified versus what remains unverified.
