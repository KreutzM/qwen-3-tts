"""Manual GPU benchmark worker: one loaded SDPA model and reusable clone prompt."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .llama_runner import audio_info
from .metrics import ResourceMonitor, real_time_factor


def run_batch(args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from .runtime import DEFAULT_MODEL_ID

    revision = args.revision
    snapshot = snapshot_download(DEFAULT_MODEL_ID, revision=revision, local_files_only=True)
    codec = json.loads((Path(snapshot) / 'speech_tokenizer/config.json').read_text())
    frame_duration = codec['decode_upsample_rate'] / codec['output_sample_rate']
    load_start = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(str(snapshot), device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_start
    prompt_start = time.monotonic()
    transcript = args.transcript.read_text(encoding="utf-8").strip() if args.mode == "transcript" else None
    if args.mode == "transcript" and not transcript:
        raise ValueError("Exact reference transcript must be non-empty.")
    prompt = model.create_voice_clone_prompt(ref_audio=str(args.reference), ref_text=transcript, x_vector_only_mode=args.mode == "speaker")
    torch.cuda.synchronize()
    prompt_seconds = time.monotonic() - prompt_start
    rows = []
    manifest = {"backend": "python", "model_id": DEFAULT_MODEL_ID, "model_revision": revision,
                "attention_backend": "sdpa", "dtype": "float16", "model_load_s": load_seconds,
                "reference_prompt_s": prompt_seconds, "frame_duration_s": frame_duration, "frame_limit": args.frames, "language": "de", "conditioning": "speaker_only" if args.mode == "speaker" else "audio_and_transcript"}
    for text_path in sorted(args.texts.glob('*.txt')):
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty benchmark passage: {text_path.name}")
        for repetition in range(args.runs):
            torch.manual_seed(args.seed + repetition)
            torch.cuda.reset_peak_memory_stats()
            monitor = ResourceMonitor(os.getpid())
            monitor.start()
            row = manifest | {"passage": text_path.stem, "run": repetition + 1, "input_characters": len(text),
                              "seed": args.seed + repetition, "sampling": {"temperature": 0.9, "top_k": 50, "top_p": 1.0,
                                  "do_sample": True, "subtalker_temperature": 0.9, "subtalker_top_k": 50, "subtalker_top_p": 1.0, "subtalker_dosample": True},
                              "timing_scope": "warm process; first synthesis identified separately", "first_synthesis": len(rows) == 0,
                              "listening_assessment": "pending", "status": "error", "process_rtf": None}
            started = time.monotonic()
            fatal = False
            try:
                waves, rate = model.generate_voice_clone(text=text, language="German", voice_clone_prompt=prompt,
                    max_new_tokens=args.frames, do_sample=True, temperature=0.9, top_k=50, top_p=1.0,
                    subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
                torch.cuda.synchronize()
                row['synthesis_s'] = time.monotonic() - started
                output = args.destination / f'{text_path.stem}-{repetition+1}.wav'
                # The parent allocates an exclusive destination; refuse any unexpected existing file.
                with output.open('xb') as handle:
                    sf.write(handle, waves[0], rate, format='WAV')
                row.update(audio_info(output))
                # Audio duration alone cannot establish EOS in the Python API; conservative cap check.
                if row['audio_duration_s'] >= args.frames * frame_duration - 0.25:
                    raise ValueError('Output is at the nominal frame cap; termination needs inspection.')
                row['synthesis_rtf'] = real_time_factor(row['synthesis_s'], row['audio_duration_s'])
                row['status'] = 'ok'
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
                # Preserve partial results and terminate on OOM instead of repeating a failing allocation.
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    fatal = True
            finally:
                row.update(monitor.finish())
                row['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                row['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            rows.append(row)
            (args.destination / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')
            if fatal:
                raise RuntimeError('CUDA OOM; partial benchmark results were preserved.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('speaker', 'transcript'), required=True)
    for field in ('reference', 'transcript', 'texts', 'destination'):
        parser.add_argument(f'--{field}', type=Path, required=field != 'transcript')
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    run_batch(args)


if __name__ == '__main__':
    main()
