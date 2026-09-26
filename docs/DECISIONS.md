# Project decisions

## ADR-001 — WSL2 is the primary runtime

**Decision:** Use Windows 11 as host and WSL2/Ubuntu as the ML runtime.

**Rationale:** Linux is the better-supported environment for the Qwen/PyTorch ecosystem and optional FlashAttention, while WSL2 can use the Windows-managed NVIDIA GPU driver.

## ADR-002 — consume upstream; do not fork Qwen3-TTS

**Decision:** This repository is an experiment/integration workspace, not a fork of `QwenLM/Qwen3-TTS`.

**Rationale:** It keeps local code small and makes upstream upgrades explicit. Use the official `qwen-tts` package/model IDs unless a reproducible blocker requires otherwise.

## ADR-003 — 1.7B Base is the first target

**Decision:** Start with `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and German voice cloning.

**Rationale:** The primary goal is to assess German voice-cloning quality on the RTX 3060 12 GB before optimizing or comparing variants.

## ADR-004 — SDPA baseline before FlashAttention

**Decision:** Prove a standard PyTorch attention path first; add FlashAttention 2 later as an optional optimization.

**Rationale:** This separates functional problems from CUDA build-toolchain problems and gives a measurable baseline for VRAM/speed comparisons.

## ADR-005 — no private/generated audio in Git

**Decision:** Reference recordings and generated WAV files remain local.

**Rationale:** Voice samples can be sensitive/identifying and model/audio artifacts are large. Git should contain code, metadata and compact benchmark summaries only.

## Pending decisions

- Public repository license for this integration code (owner decision; do not assume one).
- Whether long-form generation should use the Python API directly or a service layer.
- Forced-alignment stack for future EPUB/DAISY Media Overlays.

## ADR-006 — evaluate llama.cpp as an optional backend in this repository

**Decision:** Develop on `feat/llama-cpp-backend`, consuming the official pinned
llama.cpp source and ggml-org Q8_0 conversion alongside the existing Python
SDPA implementation. Use project-local compiler tools where system tools are
missing. Keep the original Python dependency environment intact.

**Rationale:** Both backends share the target model, machine, consented reference
recording and German test texts. A separate repository would duplicate the
benchmark and reproducibility records. Upstream provides native `llama-tts`
support, so a community fork is unnecessary for the initial experiment.

**Status:** Evaluation complete on 2026-09-26. Retain llama.cpp as an optional
local backend, with Python SDPA as the established reference and fallback.
Pinned build/rerun, artifact integrity and actual consented cloning are verified.
The matched three-mode comparison completed all 27 runs without failures. The
user accepted one sample per passage and mode and reported no audible difference
between Python and llama.cpp. Native reported synthesis RTF is 0.378–0.394;
native fresh-process RTF is 0.526–0.596; Python generation RTF is 2.209–2.380
(medians by passage/mode). This supports the optional backend on this machine.

Timing scopes, Q8/float16 differences, device-wide memory, subjective unblinded
listening and unreviewed repetitions limit generalization. No general quality
equivalence or universal speedup is claimed. Python dependencies and lock remain
unchanged. See [LLAMA_CPP_RESULTS.md](LLAMA_CPP_RESULTS.md) for commands, exact
pins, complete metrics, listening scope and the final adoption decision.
