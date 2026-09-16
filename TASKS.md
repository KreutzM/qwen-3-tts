# Codex task plan

This file is the execution handoff for Codex CLI. Work phase by phase and record measured facts rather than assumptions.

## Phase 1 — establish the local baseline

### 1. Inspect, do not guess

- Read `AGENTS.md`, `README.md`, `docs/SETUP_WSL2.md` and `docs/DECISIONS.md`.
- Run `./scripts/doctor.sh` before installing anything.
- Confirm the repository is in the WSL Linux filesystem where practical.
- Capture only non-sensitive environment facts needed for reproducibility.

### 2. Bootstrap the Python environment

Run `./scripts/bootstrap.sh` and resolve concrete failures with the smallest necessary change.

Important constraints:

- Do not install a Linux NVIDIA display driver in WSL.
- Do not add FlashAttention until the normal PyTorch path works.
- Prefer the official `qwen-tts` package and Python 3.12.
- If PyTorch installs without CUDA support, replace it with the current compatible CUDA-enabled PyTorch build using official PyTorch guidance, then document the exact choice.

### 3. Verify the environment

With `.venv` activated:

```bash
pytest -q
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Create/update `docs/ENVIRONMENT.md` with the validated versions and backend. Do not include usernames, hostnames or secrets.

### 4. Verify Qwen3-TTS model loading

Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Start with the default/SDPA path. Confirm it loads on the RTX 3060 12 GB without OOM.

A model download is expected and must remain outside Git.

### 5. Run a German clone smoke test when reference data exists

Expected local files:

- `local_data/reference.wav`
- `local_data/reference.txt` — exact transcript of the reference recording

If both exist, run a short clone using `qwen3-clone`. Write output to `outputs/`.

If they do not exist, stop this sub-step cleanly and report that a user-owned reference sample is still needed. Do not invent or download a person's voice for cloning.

### 6. Freeze the working software state

Once the baseline is proven:

```bash
python -m pip freeze --exclude-editable > requirements.lock.txt
```

Review the lock file before committing it. Record any special PyTorch install index/command in `docs/ENVIRONMENT.md`, because `pip freeze` alone may not encode the wheel source.

### 7. Commit the validated baseline

Update this task file with completion notes or checkboxes only for work actually verified. Keep environment fixes and optional optimizations distinguishable in history.

## Phase 2 — characterize German quality and performance

Create a benchmark runner that uses the same reusable voice-clone prompt for all test passages and records at least:

- model ID
- attention backend
- input character count
- generated audio duration
- wall-clock generation time
- real-time factor (generation time / audio duration)
- peak allocated/reserved CUDA memory
- output sample rate

Use the texts under `experiments/texts/`. Store compact JSON/CSV/Markdown results in Git; keep generated WAV files in ignored `outputs/`.

Run each passage more than once so cold-start/model-load time is not confused with steady-state synthesis time.

## Phase 3 — evaluate FlashAttention and model variants

Only after Phase 1 is stable:

- Try FlashAttention 2 with a compatible CUDA/PyTorch toolchain.
- Compare it to SDPA on identical inputs.
- Optionally compare 1.7B Base vs 0.6B Base for speed/quality trade-offs.
- Do not keep an optimization that is fragile and unmeasured.

## Phase 4 — long-form / audiobook foundation

Investigate paragraph/chapter chunking, deterministic file naming, resume/retry behavior, text normalization for German numbers/units/abbreviations, and later forced alignment for EPUB/DAISY Media Overlays.

Do not implement DAISY packaging until the TTS baseline and long-form segmentation are characterized.
