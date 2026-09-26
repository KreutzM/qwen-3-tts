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

## llama.cpp preparation (2026-09-26)

Measured on the target workstation; synthesis results are recorded separately.

| Item | Measured / selected value |
| --- | --- |
| Distribution | Ubuntu 22.04.5 LTS |
| WSL2 kernel | 6.18.33.2-microsoft-standard-WSL2 |
| CPU reported by WSL | AMD Ryzen 9 5950X, 32 logical CPUs, x86_64 |
| WSL RAM reported by `free -h` | 62 GiB |
| GPU | RTX 3060, 12288 MiB |
| Windows NVIDIA driver exposed in WSL | 591.86 |
| Host C++ compiler | GCC 11.4.0 |
| CMake | 3.31.6, isolated `.tools/cmake-venv` |
| Ninja | 1.11.1.4, isolated `.tools/cmake-venv` |
| CUDA compiler | 12.8.93, CUDA 12.8.1 component manifest |
| llama.cpp commit | `81bc6b83f827df746eb129235488d325c49cae52` |
| GGUF conversion repository | `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` |
| GGUF revision | `ca27d74bc954b73dadab5b71ca265d87fc861a7c` |
| Original model | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

`./scripts/doctor.sh` passed before environment changes. Python 3.12.14,
qwen-tts 0.1.1, PyTorch 2.5.1+cu121 and CUDA availability were rechecked; the
existing two unit tests passed. No Python baseline dependency version was changed.
CMake and nvcc were absent initially. Noninteractive sudo is unavailable, so
compiler tools were installed locally without a Linux display driver.

Run `./scripts/setup_llama_toolchain.sh` to reproduce the optional local toolchain.
It creates a separate Python environment for pinned CMake/Ninja, verifies the
official NVIDIA archives in `config/cuda_toolchain.json`, and assembles nvcc,
cudart, CCCL and cuBLAS under `.tools/cuda-12.8.1/`. Component licenses remain
in `.tools/cuda-12.8.1.staging/components/`. This follows NVIDIA's documented
[component archive approach](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables).
It installs no system packages and does not modify shell startup files. A failed
incomplete toolkit must be moved aside explicitly before retrying.

`./scripts/bootstrap_llama_cpp.sh` pins the official upstream checkout, refuses
dirty/unexpected remotes, and configures Release, `GGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=86`, the local toolkit root/compiler,
`LLAMA_OPENSSL=OFF`, and `LLAMA_BUILD_TESTS=OFF`, with eight build jobs by default.
HTTPS downloads are handled by the separate checksummed artifact downloader;
the binary is used with explicit local files. Upstream's own test suite is not
part of this project's model-free tests.

The selected SHA is ahead of both the CPU graph-order fix (`c8e03ce...`, PR
#26649) and text-stream/EOS fix (`217df17...`, PR #26706), verified with the
GitHub compare API. The selected source documents German, speaker references,
and separate talker/mmproj inputs. Actual build and synthesis acceptance must
still be checked; source inspection is not runtime evidence.

Run `./scripts/download_llama_models.sh` for only the selected Q8_0 talker and
audio companion. Both downloads completed and passed the pinned size/checksum
checks. Full checksums and byte sizes are in `config/llama_model.json`; artifact
files remain under ignored `models/qwen3-tts-llama/`. File size is not VRAM use.
There is still no local reference WAV/transcript pair.

The first CMake compiler-identification attempt failed because nvcc expected
`lib64/` while the component archives used `lib/` (`-lcudadevrt` and
`-lcudart_static` not found). The local setup now creates `lib64 -> lib` when
absent. This fixes directory layout without changing any dependency version.
The failed build log is retained locally for comparison.

A second build reached final linking but failed because the non-system CUDA
shared libraries were omitted from the generated runtime search path
(`libcudart.so.12` / `libcublas.so.12` not found). Bootstrap now passes
`CMAKE_BUILD_RPATH=<detected-toolkit>/lib64`, keeping toolkit resolution local.
Compiled objects are reused; no upstream source or dependency version is changed.

## llama.cpp runtime verification (2026-09-26)

- Bootstrap completed successfully after the documented local library-layout
  and RPATH fixes. A second successful bootstrap reused the existing build.
- `llama-tts --version` reports 0.5.0-dev, commit 81bc6b8, GCC 11.4.0.
- `ldd` resolves libcudart/cuBLAS from the local toolkit and libcuda from WSL.
- Final smoke: 51 frames, 4.08 s, 24000 Hz, non-silent, regular termination below
  the 300-frame cap; 5.279 s process wall time / 1.91 s reported synthesis time.
- Trace logs: 29/29 talker layers offloaded; two audio-generation contexts use
  CUDA0. Talker Flash Attention explicitly off; context explicitly 4096. Small
  CPU buffers and CPU-mapped model data remain. Speaker encoding is unverified.
- Nine native benchmark runs (three per existing German passage) completed
  without OOM, crashes, silent output or frame-cap termination. Median total
  process RTF 0.472–0.527, synthesis RTF 0.362–0.372. GPU memory figures are
  sampled device-wide usage, including other workloads, not exclusive model use.
- The user accepted the short smoke and one sample per German benchmark passage
  by listening on 2026-09-26 ("alle ok"). Consent-dependent cloning and Python
  comparison were pending at this checkpoint; consented input was supplied later
  (see the native cloning update below).
  Unit fixtures are not real GPU cloning evidence.

See [LLAMA_CPP_RESULTS.md](LLAMA_CPP_RESULTS.md) and its sanitized result links.
`requirements.lock.txt` retained SHA-256
`8a6ab989175719fe1a7c008f67607d8871d1c6386234d43cf8b2452e73208e7a`;
Python qwen-tts/PyTorch versions and the baseline tests remain unchanged.

The existing Python `load_model(..., attention="sdpa")` helper was also re-run
against the cached pinned local snapshot after the llama.cpp installation. It
loaded successfully on the GPU with its unchanged preferred bfloat16 dtype:
4,203,216,384 bytes peak allocated and 4,334,813,184 bytes peak reserved. This
checks model loading; it is not a clone benchmark. The comparison worker uses
explicit float16 as documented, rather than silently conflating these profiles.

An initial offline probe using the repository ID failed because the installed
Transformers tokenizer's `fix_mistral_regex` path queried model metadata despite
`HF_HUB_OFFLINE=1`. Using the existing cached snapshot's local path avoided that
lookup and passed. No package versions or original runtime helper were changed.
The new Python benchmark worker already uses this local-snapshot approach.

The exact comparison-worker load profile was separately checked against the
same cached revision: Qwen3TTSModel with explicit float16, SDPA and cuda:0 loaded
successfully offline. Reference-prompt creation and actual clone generation
still require consented input and have not been measured.

### Consented native cloning smoke (2026-09-26)

The user supplied reference audio/transcript, confirmed usage permission and
confirmed timestamp alignment. A private 15-second mono 24 kHz reference
produced a native clone of 5.12 seconds / 64 frames without OOM or cap failure.
The user accepted speaker similarity and sentence quality by listening
("ja, alles ok"). Detailed measurements and preparation scope are recorded
in [LLAMA_CPP_RESULTS.md](LLAMA_CPP_RESULTS.md). The subsequent full conditioned comparison and user review completed; see
the final evaluation below.

### Final matched backend evaluation (2026-09-26)

- Shared 13-second mono 24 kHz consented reference, exact timestamp-selected
  transcript, same three passages, three seeds/runs per mode: 27/27 successful.
- Modes: native Q8_0 speaker-only, Python float16/SDPA speaker-only and Python
  float16/SDPA audio-plus-transcript. No package/lock changes.
- User accepted repetition 1 of every passage/mode (nine samples), including
  speaker similarity and sentence quality; no audible native/Python difference
  was reported. Other repetitions have runtime/structural evidence only.
- Median native synthesis RTF 0.378–0.394, fresh-process RTF 0.526–0.596;
  Python generation RTF 2.209–2.380. Load/prompt times are separately recorded.
- GPU experiments were serialized; user confirmed no other intensive GPU apps.
  Device-wide memory still includes Windows/desktop, with limited WSL attribution.
- Final decision: retain optional llama.cpp backend; preserve Python reference
  and fallback. Detailed pins, memory/RSS, measurements and limitations:
  [LLAMA_CPP_RESULTS.md](LLAMA_CPP_RESULTS.md).
