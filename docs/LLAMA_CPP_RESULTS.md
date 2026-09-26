# llama.cpp evaluation: 2026-09-26

## Verified scope

The optional backend builds and produces structurally valid, non-silent German
WAV files from German input on the target RTX 3060 under WSL2. All nine synthesis benchmark runs
completed before the frame cap, without OOM or process failures. This is runtime
and timing evidence. On 2026-09-26, the user listened to the short smoke test
and one sample from each of the three passages and reported "alle ok" (all OK).
No intelligibility, repetition, sentence-ending or pronunciation issues were
reported for those four files. The other reference-free benchmark repetitions were not reviewed by listening.
Subsequent consented cloning and speaker-similarity reviews are recorded below.

The initial reference-free evaluation is complete. The user subsequently
supplied consented reference audio and transcript; a native cloned smoke sample
was accepted by listening (see the section below). Python conditioning
comparisons and the final adoption decision are recorded below. No synthetic
test fixture was used as a real cloning reference.

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
limit and had nonzero RMS. The user accepted this sample by listening.

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
metrics remain separate. This worker has fixture-based logic tests and its exact cached float16/SDPA
model-load profile was checked on the GPU, and its actual reference-conditioned benchmark completed all 18 Python runs
without process/OOM/frame-cap failures.

Sampling parameters and seeds are recorded; code-predictor implementations and
conditioning differ, so equal seeds do not imply equivalent samples. The native
CLI exposes talker temperature but does not expose a separate code-predictor
temperature; Python's subtalker sampling settings are explicitly recorded.

The final cloning listening check covers repetition 1 of each passage and mode:
umlauts, numbers, abbreviations, terminology, pacing, omissions, repetitions,
sentence endings and speaker similarity. Actual user observations are recorded
below; quality is not inferred from timings or WAV shape.

## Final adoption decision

Retain llama.cpp as an optional backend for local German synthesis and cloning
on this workstation; retain Python SDPA as the established reference/fallback.
The pinned Q8_0 setup and rerun work, actual cloning succeeds, all 27 matched
three-mode runs succeed, and the user accepted all nine reviewed passage
samples. Native synthesis and even fresh-process RTF are lower than Python
warm generation RTF in this evaluation. This supports offering the optional
backend; it does not prove general quality equivalence or a universal speedup.

The Python environment and dependency lock are preserved. No new service/UI,
model-variant tuning, automatic merge or FlashAttention optimization is part of
this adoption. Changing pins, speakers, texts or hardware requires a new local
evaluation. All required tracker #3 evidence is recorded; unreviewed seeds and
objective/blind quality evaluations are explicit limits, not claimed results.

## User listening evidence

The user accepted these four private outputs on 2026-09-26 with "alle ok":

- `outputs/llama-cpp/run-wgztknfh/speech.wav` (short smoke).
- `outputs/benchmark/batch-jfzv8mec/llama/run-y70a2y2d/speech.wav` (prose).
- `outputs/benchmark/batch-jfzv8mec/llama/run-b_py9zha/speech.wav` (technical).
- `outputs/benchmark/batch-jfzv8mec/llama/run-515txob5/speech.wav` (edge cases).

This is user listening feedback, not an automated quality score or a speaker
similarity assessment. At that initial checkpoint no reference recording was available. The user
subsequently supplied consented input; see the native cloning section below.

## Consented reference and native clone smoke

On 2026-09-26 the user supplied a three-minute MP3 excerpt and a timestamped
transcript from the longer recording, confirmed usage permission, and confirmed
that transcript timestamps 0–180 seconds correspond to the excerpt. Original
files remain private and unchanged. Prepared references are mono 24 kHz PCM
under ignored `local_data/`; preparation provenance stays private.

A first speaker-only smoke used the first 15 seconds. It produced 64 frames,
5.12 seconds at 24 kHz, with nonzero RMS and natural termination below the
300-frame limit. Process time was 6.531 seconds, reported synthesis time 2.04
seconds, and sampled device-wide GPU peak 8606 MiB. Private output:
`outputs/llama-cpp/run-a84iazng/speech.wav`. The user listened and answered
"ja, alles ok" to the speaker similarity, intelligibility, completeness and
repetition questions. This verifies the short cloned sample, not all benchmark
passages or the relative quality of Python conditioning modes.

For a full comparison, a separate first-13-second reference ends at the next
transcript boundary. Its exact provided transcript comprises the segments
starting at 0:01 and 0:07; timestamps and the longer document header are removed.
All configurations in that comparison use the same prepared audio.

## 15-second speaker-only preliminary comparison

The first reference-based batch completed all 18 runs (three per passage and
backend) without process/OOM/frame-cap failures. Both backends used the same
15-second mono 24 kHz reference and seeds 42/43/44. Native inference is a fresh
CLI process; Python loads once and reuses a speaker-only prompt. This batch
contains no transcript-conditioned configuration and is separate from the
13-second three-mode comparison.

| Passage | Native median synthesis RTF (range) | Python speaker-only median synthesis RTF (range) |
| --- | --- | --- |
| `de_edge_cases` | 0.369 (0.363–0.379) | 2.158 (2.121–2.238) |
| `de_prose` | 0.380 (0.377–0.390) | 2.154 (2.126–2.218) |
| `de_technical` | 0.380 (0.372–0.390) | 2.266 (2.243–2.409) |

Python model loading took 4.380 seconds and reference-prompt creation 1.554
seconds, separate from per-passage synthesis. These are measured results on
this workstation, not equivalent-conditioning or quality guarantees. GPU runs
were sequential; device memory includes unrelated Windows/desktop usage, which
was not terminated. WSL compute-app enumeration returned no attributable
process rows. The user confirmed no other GPU-intensive applications were running during
the measurements. Experiments are serialized; Windows/desktop allocations
remain included and WSL cannot establish process-exclusive GPU usage.

Compact measurements: [rows](../experiments/results/llama_cpp/2026-09-26-reference-speaker-only/runs.json)
and [summary](../experiments/results/llama_cpp/2026-09-26-reference-speaker-only/summary.json).
Private outputs: `outputs/benchmark/batch-8hox9qj8/`. The subsequent full three-mode evaluation and listening review are recorded
below.

## Full three-mode comparison: shared 13-second reference

All 27 runs completed successfully: three repetitions per passage for native
Q8_0 speaker-only, Python float16/SDPA speaker-only, and Python float16/SDPA
audio-plus-transcript conditioning. The same prepared reference, German texts,
seeds 42/43/44, 1200-frame bound and 600-second nominal run bound were used.
The Python process timeout applies to the whole nine-run batch. All outputs
were readable, non-silent 24 kHz WAVs ending below the nominal duration/frame
cap. Natural native frame counts were available in the upstream log.

Reproduce with the privately prepared files:

```bash
./scripts/benchmark_tts.sh --reference local_data/reference-first13s.wav \
  --transcript local_data/reference-first13s.txt --runs 3 \
  --export experiments/results/llama_cpp/<new-three-mode-run>
```

| Mode | Passage | Runs / failures | Median synthesis RTF (range) | Median total native process RTF (range) | Sampled GPU peak range (MiB) |
| --- | --- | --- | --- | --- | --- |
| llama.cpp Q8 speaker-only | `de_edge_cases` | 3 / 0 | 0.394 (0.390–0.397) | 0.573 (0.534–0.574) | 8529–8529 |
| llama.cpp Q8 speaker-only | `de_prose` | 3 / 0 | 0.378 (0.376–0.389) | 0.596 (0.573–0.599) | 8527–8527 |
| llama.cpp Q8 speaker-only | `de_technical` | 3 / 0 | 0.378 (0.374–0.379) | 0.526 (0.523–0.545) | 8529–8529 |
| Python SDPA speaker-only | `de_edge_cases` | 3 / 0 | 2.257 (2.244–2.277) | not measured (persistent worker) | 9343–9574 |
| Python SDPA speaker-only | `de_prose` | 3 / 0 | 2.262 (2.251–2.269) | not measured (persistent worker) | 9560–9792 |
| Python SDPA speaker-only | `de_technical` | 3 / 0 | 2.298 (2.249–2.338) | not measured (persistent worker) | 10022–11011 |
| Python SDPA audio + transcript | `de_edge_cases` | 3 / 0 | 2.380 (2.291–2.410) | not measured (persistent worker) | 9684–11053 |
| Python SDPA audio + transcript | `de_prose` | 3 / 0 | 2.265 (2.255–2.266) | not measured (persistent worker) | 9682–9689 |
| Python SDPA audio + transcript | `de_technical` | 3 / 0 | 2.209 (2.206–2.256) | not measured (persistent worker) | 9616–9837 |

Python load and prompt creation are separate from generation:

| Conditioning | Model load | Prompt creation |
| --- | --- | --- |
| `speaker_only` | 4.661 s | 0.840 s |
| `audio_and_transcript` | 3.031 s | 0.897 s |

The first synthesis is labelled per Python mode. The table reports all three
runs, including that first call; subsequent warm calls remain individually
identifiable in the exported rows. Native timing is fresh-process; its rounded
upstream component total differs in scope/precision from Python timed API
generation. Even native total-process RTF is lower in these samples than Python
generation RTF, but this is a workstation result, not a general backend ratio.
Generated durations and exact samples differ despite equal seeds.

GPU experiments were serialized. The user confirmed no other GPU-intensive
applications were running; Windows/desktop allocations were retained. Identical
0.5-second device-level sampling covers both backends. Peak host process-tree
RSS and separate PyTorch allocated/reserved CUDA counters are in the exports;
sampled device peaks include background memory and may miss transients.
Native logs show 29/29 talker layers, audio encoder and audio generation contexts
on CUDA0, with small CPU compute buffers and CPU-mapped data remaining. Python
loads the model explicitly on cuda:0 with SDPA; file/audio preprocessing remains
on CPU. No all-GPU execution or model-exclusive memory claim is made.

Compact [JSON rows](../experiments/results/llama_cpp/2026-09-26-reference-three-modes/runs.json),
[CSV rows](../experiments/results/llama_cpp/2026-09-26-reference-three-modes/runs.csv),
and [summary](../experiments/results/llama_cpp/2026-09-26-reference-three-modes/summary.json)
exclude private voice/transcript/path data. Raw results/audio/logs stay in
`outputs/benchmark/batch-tm4jaw2f/`. Generation-time listening fields remain
`pending`; later user listening evidence is recorded separately below.

### Listening scope

The user accepted repetition 1 of all three native passages ("sie sind ok")
and reported hearing no difference between Python and llama.cpp. The user also
explicitly accepted repetition 1 of the three Python speaker-only passages
("sind ok"). Questions covered intelligibility, completeness, repetition,
speaker similarity and pronunciation of numbers, umlauts and abbreviations.
No problems were reported. The user subsequently accepted repetition 1 of all three Python
transcript-conditioned passages ("sind ok") in response to the same quality
and similarity questions. No additional differences or defects were reported.
This is an unblinded subjective review of nine samples, not proof that all
seeds or speakers have equal quality. Repetitions 2 and 3 remain unreviewed
by listening; their runtime and structural checks passed.
