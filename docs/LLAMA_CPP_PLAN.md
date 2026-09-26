# Plan: Qwen3-TTS with llama.cpp

Research date: 2026-09-26. Status: proposed implementation; no llama.cpp build,
model download or synthesis was performed while writing this plan.

## Objective and recommendation

Evaluate German synthesis and voice cloning with the official
`ggml-org/llama.cpp` implementation on the project's WSL2 workstation
(Ryzen 9 5950X, 64 GB RAM, RTX 3060 with 12 GB VRAM). Use
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` through its GGUF conversion, starting with Q8_0.
Keep the official `qwen-tts` SDPA environment as the reference and fallback.

Native support is available: upstream merged Qwen3-TTS in PR #26254 on
2026-08-04. The dedicated executable is `llama-tts`, using `libmtmd` for audio
processing. A community fork or a new model port is therefore unnecessary for
the first experiment. [Upstream implementation](https://github.com/ggml-org/llama.cpp/pull/26254)

The existing [environment record](ENVIRONMENT.md) documents successful GPU model
loading on 2026-09-16 with Python 3.12.14, qwen-tts 0.1.1 and PyTorch
2.5.1+cu121. It does not demonstrate a completed voice clone; no reference pair
was available at that time. Treat these as historical measurements and recheck
the machine before implementation. This experiment extends the characterization
work in [TASKS.md](../TASKS.md).

## Supported path and important boundaries

The upstream TTS tool documents German via `--tts-lang de`, reference audio via
`--tts-speaker-file`, and WAV output via `--output`. Its `-n` limit counts audio
frames. Use the dedicated tool's documentation rather than generic Hugging Face
examples for text generation. [TTS usage](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md)

The inference path is:

```text
German text + optional local reference recording
    -> llama-tts
    -> talker GGUF + matching mmproj GGUF
    -> speech code generation and waveform decoding
    -> outputs/llama-cpp/<run-id>/speech.wav
```

The talker and audio components are both required. The upstream implementation
uses a speaker encoder, talker, code predictor and waveform decoder; converting
only a normal Qwen text model cannot supply this pipeline. The original support
PR treated server integration as follow-up work. Start with the CLI and verify
server capabilities separately at the selected revision before planning an API.
[Implementation design](https://github.com/ggml-org/llama.cpp/pull/26254)

Do not assume parity with transcript-conditioned Python cloning. The documented
llama.cpp interface accepts speaker audio without a reference transcript. Python
also offers `x_vector_only_mode=True`, which uses speaker embedding alone and
may reduce cloning quality. Compare equivalent conditioning where possible and
report the full Python audio-plus-transcript mode separately.
[Qwen voice-cloning API](https://github.com/QwenLM/Qwen3-TTS#voice-clone)

## Stage 1: inspect the workstation and select a revision

Before any environment changes, run from the repository root:

```bash
./scripts/doctor.sh
nvidia-smi
cmake --version
c++ --version
nvcc --version
```

Record the actual Ubuntu/WSL version, compiler, CMake, CUDA toolkit, Windows
driver exposed through WSL and available GPU memory. A missing `nvcc` means a
build prerequisite is missing even if PyTorch CUDA works. PyTorch's packaged
runtime and the CUDA compiler toolkit are separate installations.

If necessary, install a compatible toolkit following NVIDIA's WSL guidance.
Never install a Linux NVIDIA display driver or a meta-package that brings one
into WSL. Select versions from the detected driver/toolchain rather than
assuming the historical PyTorch CUDA 12.1 value dictates the compiler version.
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Choose an upstream release or full commit SHA containing Qwen3-TTS support and
subsequent relevant fixes. Record the SHA before building; do not make a moving
`master` branch the reproducibility specification. Check ancestry for the fixes
linked below, and inspect that revision's TTS help and model instructions.

Deliverable: toolchain inventory and selected llama.cpp revision in
`docs/ENVIRONMENT.md`, clearly marked as measured only after execution.

## Stage 2: build an isolated CUDA executable

Use the already ignored `.tools/llama.cpp/` for the upstream checkout. A future
setup script should clone only when absent, validate the remote, refuse to
discard local edits, and explicitly check out the selected SHA. Keep generated
build files there and avoid changing `.venv` or `requirements.lock.txt`.

After checkout, run from this repository's root:

```bash
cmake -S .tools/llama.cpp -B .tools/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build .tools/llama.cpp/build --config Release \
  --target llama-tts -j 8
.tools/llama.cpp/build/bin/llama-tts --help
```

Confirm the target and options against the pinned source. Inspect runtime logs
for CUDA device selection and actual offloading; successful compilation alone
does not prove GPU inference. Record compiler versions and CMake options.
[Official build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

## Stage 3: obtain and pin the GGUF artifacts

Use the conversion published by `ggml-org`, distinguishing it from the original
Qwen checkpoint. Download only the selected pair into ignored
`models/qwen3-tts-llama/`:

| Experiment | Talker file | Audio companion |
| --- | --- | --- |
| Initial baseline | `Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` | `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` |
| Later memory comparison | `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` | Same Q8_0 companion |
| Optional quality control | `Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` | `mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` |

The listed Q8_0 files total approximately 2.30 GB on disk; this does not predict
peak VRAM. Record the model repository's full revision, filenames, sizes and
SHA-256 checksums. Use revision-specific downloads and explicit local paths for
repeatable runs. Verify the pairing at the selected code revision.
[GGUF artifact repository](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/tree/main)

If conversion becomes necessary, use the selected upstream converter with the
complete original checkpoint and its audio assets. Inspect its supported
arguments first, use an isolated conversion environment, and record the source
model revision and conversion commands. Preconverted artifacts are the initial
route; custom conversion is a separate troubleshooting task.

## Stage 4: run bounded German smoke tests

The following is an implementation template, not a locally validated command.
Run after Stages 1–3, from the repository root, with a fresh output directory:

```bash
run_dir="outputs/llama-cpp/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
.tools/llama.cpp/build/bin/llama-tts \
  -m models/qwen3-tts-llama/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj models/qwen3-tts-llama/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  -ngl 99 --tts-lang de -n 300 \
  -p 'Guten Tag. Dies ist ein kurzer Test der deutschen Sprachausgabe.' \
  --output "$run_dir/speech.wav"
```

Execution note (2026-09-26): the selected codec uses 1920 samples per frame at
24000 Hz, giving 12.5 frames per second despite the 12Hz model name. A 300-frame
cap therefore corresponds to approximately 24 seconds. Check that synthesis ends naturally before the cap;
truncation at the cap is a failure for this short sentence. Add a process timeout
in the eventual runner. Record exit status and logs, inspect the WAV's duration,
sample rate and non-silent content, and listen for intelligibility and repetition.

For a cloning run, add `--tts-speaker-file local_data/reference.wav` only when a
user-owned or consented sample exists. Use a new output directory. Retain an
exact transcript in `local_data/reference.txt` for the Python comparison; do not
invent a llama.cpp transcript flag. Never overwrite the reference. If no sample
exists, record cloning as unverified and complete only the synthesis smoke test.

On OOM, preserve the failing configuration and GPU memory observations before
trying fewer offloaded layers or Q4_K_M. Inspect audio-component placement too;
the talker layer setting alone is insufficient evidence that every component is
on the GPU.

## Stage 5: compare quality and performance

Use the existing German texts in `experiments/texts/` with identical reference
audio and text across backends. Run each passage at least three times per
configuration, sequentially, with competing GPU processes stopped where practical.

Compare these configurations:

1. Official Python SDPA with speaker-only conditioning, if supported by the
   installed version, for the closest available conditioning comparison.
2. Official Python SDPA with reference audio and transcript as the quality reference.
3. llama.cpp CUDA with Q8_0 talker and Q8_0 companion.
4. Q4_K_M or BF16 only after configuration 3 passes the smoke test.

Record input length, conditioning mode, revisions, quantization, sampling
parameters, seed where supported, output sample rate, audio duration, process
wall time, synthesis time when independently measurable, and RTF
(`synthesis seconds / audio seconds`). Equal seeds do not imply equal samples
across runtimes. Report median and range, plus every failure.

Separate model loading from generation. A fresh CLI process reloads the model;
repeated CLI launches are not proof of warm in-process inference. Label total
process RTF separately when the generation interval cannot be isolated.

Sample GPU memory with timestamps using available NVIDIA tooling and report
measurement limitations under WSL. Compare device-level memory using the same
method for both backends; Python allocated/reserved CUDA counters are not
equivalent to total device usage. Also record peak host RAM and CPU fallbacks.

Listen for pronunciation of umlauts, numbers, abbreviations and technical terms,
speaker similarity, pacing, omissions, repeated phrases and sentence endings.
Do not infer an improvement from lower file size or tokens per second.

Deliverables: compact, non-sensitive JSON/CSV results under
`experiments/results/llama_cpp/` and an English comparison document under `docs/`.
Audio and detailed logs stay under ignored `outputs/`. Remove identifiers and
private transcript content before committing summaries.

## Stage 6: integrate only after validation

Proposed implementation work, in order:

- Add an idempotent `scripts/bootstrap_llama_cpp.sh` with an explicit revision.
- Add a small `scripts/llama_tts.sh` or Python subprocess wrapper with model-path
  checks, German defaults, timeout, unique output paths and actionable errors.
- Add download-free tests for wrapper argument construction, missing files,
  timeout/error propagation and overwrite protection. Mock the process boundary.
- Add a manual benchmark entry point and document commands in README/TASKS.
- Record the adoption decision and measured limitations in `docs/DECISIONS.md`.

Keep the existing Python CLI/UI usable throughout. A web UI or persistent service
is a later task after checking current upstream endpoint and streaming support.

## Risks and acceptance criteria

Early support had a CPU graph failure addressed by
[PR #26649](https://github.com/ggml-org/llama.cpp/pull/26649), and a reported phrase
repetition/EOS problem is closed with
[PR #26706](https://github.com/ggml-org/llama.cpp/pull/26706). These motivate
revision selection and regression checks; they do not establish that today's
CUDA build fails on this workstation.

The experiment is complete when:

- A pinned build and pinned model pair reproduce German WAV generation on the
  actual RTX 3060 without OOM, crashes or unintended truncation.
- Logs demonstrate actual CUDA use and identify any CPU fallback.
- Cloning is assessed with consented reference audio, or explicitly remains
  pending because no such sample exists.
- Quality and performance measurements support a documented retain/reject
  decision; no speedup or memory saving is claimed without measurements.
- All applicable wrapper tests pass and reproducibility records are updated.

If stability or quality is inadequate, retain the official Python baseline and
document the smallest reproduction before evaluating other implementations.

## Verification of this plan

Project intent, recorded baseline, ignored artifact paths, upstream TTS support,
artifact names and build guidance were reviewed. Documentation changes were
checked with `git diff --check`. Toolchain compatibility, build success, GPU
memory use, audio quality and performance remain to be verified during execution.
