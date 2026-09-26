# llama.cpp evaluation: 2026-09-26

## Verified scope

The optional backend builds and produces structurally valid, non-silent German
WAV files on the target RTX 3060 under WSL2. All nine synthesis benchmark runs
completed before the frame cap, without OOM or process failures. This is runtime
and timing evidence; pronunciation, sentence completeness and speaker similarity
have not been assessed by listening.

There is no consented reference WAV/transcript pair in `local_data/`. Python
speaker-only and transcript-conditioned comparisons, llama.cpp voice cloning,
and the final quality/adoption decision remain pending. No synthetic test fixture
was used as a real cloning reference.

## Reproducibility

- llama.cpp SHA: `81bc6b83f827df746eb129235488d325c49cae52`.
- Model conversion: `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`, revision
  `ca27d74bc954b73dadab5b71ca265d87fc861a7c`, Q8_0 talker and Q8_0 mmproj.
- Original Qwen checkpoint for the future Python comparison:
  `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, cached revision
  `fd4b254389122332181a7c3db7f27e918eec64e3`.
- GCC 11.4.0, CMake 3.31.6, CUDA compiler 12.8.93; Windows driver 591.86.
- Release build, CUDA enabled, compute capability 8.6; explicit local toolkit
  runtime search path. See [ENVIRONMENT.md](ENVIRONMENT.md) for installation and
  the two reproduced directory/search-path failures fixed during setup.
- Final runner: German, 4096 context, talker Flash Attention off, 99 GPU layers,
  top-k 50, top-p 1.0, temperature 0.9. Seeds 42/43/44 per passage. No reference.
- Benchmark cap: 1200 frames, timeout: 600 seconds per native process.

The selected codec has 1920 samples per frame at 24000 Hz: 0.08 seconds per
frame (12.5 Hz). The model's 12Hz name is rounded. Recorded native frame counts
and WAV durations match this ratio. The Python worker reads the codec's pinned
configuration for its conservative duration-cap check.

Run the measured synthesis-only benchmark from the project root:

```bash
./scripts/benchmark_tts.sh --backends llama --runs 3 \
  --export experiments/results/llama_cpp/<new-run-name>
```

Use a new export directory; existing results are never overwritten. The measured
[JSON rows](../experiments/results/llama_cpp/2026-09-26-q8-no-reference/runs.json),
[CSV rows](../experiments/results/llama_cpp/2026-09-26-q8-no-reference/runs.csv),
and [aggregated results](../experiments/results/llama_cpp/2026-09-26-q8-no-reference/summary.json)
contain sanitized metrics. Private audio/argv/logs remain in ignored outputs.

## Timing and resource results

RTF is elapsed generation time divided by audio duration; a smaller value is
faster. Each native run is a fresh CLI process. Process RTF includes model
loading, initialization and output writing. Synthesis RTF uses upstream's
rounded prompt-evaluation + generation + vocoder timings, excluding startup.
No native run is labelled as warm in-process inference.

| Passage | Runs / failures | Audio duration range | Median process RTF (range) | Median synthesis RTF (range) | Sampled peak total GPU memory |
| --- | --- | --- | --- | --- | --- |
| `de_edge_cases` | 3 / 0 | 23.68–27.68 s | 0.472 (0.451–0.478) | 0.362 (0.362–0.369) | 8138–8148 MiB |
| `de_prose` | 3 / 0 | 15.84–18.16 s | 0.527 (0.514–0.529) | 0.372 (0.371–0.375) | 8136 MiB |
| `de_technical` | 3 / 0 | 18.40–23.28 s | 0.496 (0.472–0.508) | 0.366 (0.366–0.369) | 8138 MiB |

Sampled process-tree RSS peaked at approximately 2.37–2.39 GB. GPU memory is
GPU-0 device-wide usage sampled approximately every 0.5 seconds, including
Windows/other workloads; transient peaks may be missed. The workstation had
roughly 4 GB of unrelated reported GPU usage before testing, and no user
processes were stopped. WSL memory attribution is limited. These measurements
are not model-exclusive allocations and cannot be compared directly with the
historical PyTorch allocated/reserved counters.

## Smoke test and actual placement

The final-config short smoke test produced 51 frames / 4.08 seconds of 24000 Hz
PCM WAV audio. Its process time was 5.279 seconds, upstream synthesis time
1.91 seconds, and sampled device-wide peak 8322 MiB. It ended below the 300-frame
limit and had nonzero RMS. Listening acceptance remains pending.

Trace-level placement logs show 29/29 talker layers offloaded to CUDA0, talker
Flash Attention disabled, a 448 MiB CUDA KV buffer, and both audio-generation
contexts using the CUDA0 backend. CPU-mapped model data and small CPU compute
buffers still exist. The reference speaker-encoder path was not exercised.
Token/control and experimental-audio warnings remain in the upstream logs;
structurally valid output does not establish their irrelevance to quality.

All 32 project unit tests pass without model downloads or GPU inference. The
existing Python SDPA helper also reloaded the cached Base snapshot on the GPU
with unchanged bfloat16 preference and historical allocated/reserved peaks.
This is a fallback load check, separate from the planned float16 benchmark.

Both `./scripts/bootstrap_llama_cpp.sh` and its successful repeat completed with
exit status 0; the repeat reused compiled objects. `ldd` resolves cudart/cuBLAS
from the local toolkit and libcuda from WSL's Windows-driver interface.

## Remaining comparison and review

With a consented recording and transcript, run:

```bash
./scripts/benchmark_tts.sh --reference local_data/reference.wav \
  --transcript local_data/reference.txt --runs 3 \
  --export experiments/results/llama_cpp/<new-comparison-name>
```

This selects llama.cpp speaker-only, Python SDPA speaker-only and Python SDPA
with transcript conditioning. The Python worker uses the cached pinned snapshot,
float16 and one reusable prompt per conditioning mode across all passages. It
records model loading and reference-prompt creation separately, marks the first
synthesis, and samples device memory with the same monitor. PyTorch allocator
metrics remain separate. This worker has fixture-based logic tests but its real
GPU cloning path is unverified until reference input is available.

Sampling parameters and seeds are recorded; code-predictor implementations and
conditioning differ, so equal seeds do not imply equivalent samples. The native
CLI exposes talker temperature but does not expose a separate code-predictor
temperature; Python's subtalker sampling settings are explicitly recorded.

Listening should cover all three passages and a short cloned sample: umlauts,
numbers, abbreviations, terminology, pacing, omissions, repetitions, sentence
endings and speaker similarity. Record actual observations instead of inferring
quality from timings or WAV shape.

## Provisional decision

Retain the backend as an optional experiment with reproducible setup and a
bounded runner. Keep Python SDPA as the existing baseline. The observed native
runs justify continued evaluation; they do not establish an improvement over
Python or acceptable cloning quality. Final adoption remains pending in tracker
#3, children #8–#10, until reference-dependent measurements and listening review
are complete.
