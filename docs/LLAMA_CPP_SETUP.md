# Optional llama.cpp backend

Implementation tracker: [#3](https://github.com/KreutzM/qwen-3-tts/issues/3).
The [original plan](LLAMA_CPP_PLAN.md) describes acceptance and benchmark scope.
See [ENVIRONMENT.md](ENVIRONMENT.md) for measured status; commands alone are not
proof that synthesis or cloning passed on a particular machine.

## Setup

Work from the repository root inside the WSL Linux filesystem. Establish the
existing Python environment first using the normal project bootstrap; the
llama.cpp scripts use its interpreter without changing its dependencies.

```bash
./scripts/doctor.sh
# Only if the Python baseline is not installed:
./scripts/bootstrap.sh
```

Check CMake, C++, nvcc and the GPU. If compatible build tools are already present,
use them through PATH. On the validated workstation CMake/nvcc were missing, so
this optional local toolchain was used:

```bash
./scripts/setup_llama_toolchain.sh
```

It downloads checksummed NVIDIA CUDA 12.8.1 compiler/runtime/CCCL/cuBLAS archives,
creates `.tools/cuda-12.8.1/`, and installs CMake 3.31.6 / Ninja 1.11.1.4 in a
separate `.tools/cmake-venv/`. It installs no Linux NVIDIA display driver or system
packages. The Windows driver must be compatible with the selected CUDA runtime.
The script is specific to Linux x86_64; re-evaluate pins for other machines.

```bash
./scripts/bootstrap_llama_cpp.sh
./scripts/download_llama_models.sh
```

`config/llama_cpp_revision.txt` pins the official code SHA. Bootstrap refuses
unrelated/dirty checkouts, targets compute capability 8.6 for the RTX 3060, and
builds Release with CUDA and an explicit runtime search path to the toolkit libraries. Set `LLAMA_BUILD_JOBS=4` for fewer compiler processes
(allowed range 1–32). HTTPS support in the binary is disabled; downloads use the
separate artifact tool and inference uses explicit local paths.

`config/llama_model.json` pins the conversion repository revision and both Q8_0
file checksums. Files stay in ignored `models/qwen3-tts-llama/`. Concurrent
artifact downloads share an advisory lock. Interrupted downloads resume from
`.partial`; checksum failures and unexpected existing files require explicit
inspection and moving the conflicting file aside before retrying. No unknown
file is silently replaced. A failed toolkit staging directory likewise requires
inspection before retrying.

## German synthesis

```bash
./scripts/llama_tts.sh --text 'Guten Tag. Dies ist ein kurzer Test der deutschen Sprachausgabe.'
./scripts/llama_tts.sh --text-file experiments/texts/de_prose.txt --frames 1200 --timeout 600
```

The wrapper defaults to German (`de`), Q8_0 talker/mmproj, seed 42, temperature
0.9, top-k 50, top-p 1.0, 99 GPU layers, a 300-frame cap and a 300-second process
timeout. Explicit `--model`, `--mmproj`, `--executable`, `--gpu-layers` and sampling
arguments are available; use `--help`. Offload settings do not prove that all
components run on the GPU: inspect `process.log` for actual placement/fallbacks.

Each invocation creates a fresh `outputs/llama-cpp/run-*/` directory containing
`speech.wav`, `process.log` and `result.json`. These are private local artifacts.
The result includes invocation parameters, timing, audio validation and sampled
resources. Neither reference recordings nor previous outputs are overwritten.
A nonzero native exit code is propagated; timeout exits with status 124. Reaching
the frame limit is reported as a failure even if the binary wrote a WAV.
The generated-frame count must be present in the selected binary's log.

WAV validity/non-silence does not establish pronunciation, sentence completion
or speaker similarity. Listening assessment remains pending until performed.
CLI runs are fresh processes: total process time includes loading. The pinned
upstream also reports rounded prompt/generation/vocoder timings, allowing a
separate synthesis RTF. Device memory sampling is total GPU-0 usage and includes
other workloads; it is not equivalent to PyTorch allocator counters.

## Voice cloning

Only use a recording owned by the user or used with consent. Keep it under
ignored `local_data/` along with its exact transcript for Python comparison:

```text
local_data/reference.wav
local_data/reference.txt
```

```bash
./scripts/llama_tts.sh --reference local_data/reference.wav \
  --text 'Dies ist eine kurze Probe mit der Referenzstimme.'
```

The pinned llama.cpp interface uses a speaker embedding and accepts no transcript
argument. Compare it with Python speaker-only conditioning separately from the
existing audio-plus-transcript Python clone. Missing reference input leaves
cloning and reference-dependent benchmarks unverified. Never fabricate a sample
or use a generated demo as proof of real cloning.

## Checks and fallback

```bash
.venv/bin/pytest -q
bash -n scripts/setup_llama_toolchain.sh scripts/bootstrap_llama_cpp.sh \
  scripts/download_llama_models.sh scripts/llama_tts.sh
```

Unit tests use tiny fixtures and fake processes, with no model/GPU downloads.
CUDA build and real synthesis remain local/manual checks.

The existing `qwen3-clone` and `scripts/webui.sh` provide the Python backend.
Keep `.venv` and `requirements.lock.txt` unchanged while evaluating llama.cpp.
Record failures and actual versions before considering dependency changes or a
community implementation. Model variants, a new UI/service and FlashAttention
optimization are separate later decisions.
