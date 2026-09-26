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
