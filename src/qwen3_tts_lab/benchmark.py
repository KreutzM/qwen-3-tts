"""Manual cross-backend benchmark. Raw/private data stay in ignored outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from .llama_runner import ROOT, RunConfig, RunError, execute, validate
from .metrics import summarize

PUBLIC_FIELDS = {
    'backend', 'conditioning', 'passage', 'run', 'status', 'input_characters', 'language',
    'code_revision', 'model_revision', 'model_id', 'quantization', 'context', 'talker_attention',
    'attention_backend', 'dtype', 'frame_duration_s', 'frame_limit', 'sampling', 'seed', 'sample_rate', 'audio_duration_s',
    'process_wall_s', 'synthesis_s', 'process_rtf', 'synthesis_rtf', 'prompt_eval_s', 'generation_s',
    'vocoder_s', 'generated_frames', 'peak_device_used_mib', 'peak_process_tree_rss_bytes',
    'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'resource_sample_count', 'memory_method',
    'timing_scope', 'model_load_s', 'reference_prompt_s', 'first_synthesis', 'listening_assessment',
}


def public_row(row: dict) -> dict:
    """Allow only measurement fields; never export argv, text, paths or raw error strings."""
    result = {key: value for key, value in row.items() if key in PUBLIC_FIELDS}
    if row.get('status') != 'ok':
        result['failure_detail'] = 'See private local result.json/process.log; excluded from public export.'
    return result


def export_results(rows: list[dict], destination: Path) -> None:
    if not rows:
        raise ValueError('No benchmark rows to export.')
    destination.mkdir(parents=True, exist_ok=False)
    public = [public_row(row) for row in rows]
    (destination / 'runs.json').write_text(json.dumps(public, indent=2)+'\n')
    (destination / 'summary.json').write_text(json.dumps(summarize(public), indent=2)+'\n')
    fields = sorted({key for row in public for key in row})
    with (destination / 'runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in public:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def benchmark(args: argparse.Namespace) -> Path:
    texts = sorted(args.texts.glob('*.txt'))
    if not texts:
        raise ValueError('No UTF-8 benchmark passages found.')
    if args.runs < 3 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.frames <= 0 or not 0 <= args.seed <= 0xFFFFFFFF - args.runs:
        raise ValueError('Use at least three runs and positive frame/timeout bounds.')
    if len(set(args.backends)) != len(args.backends):
        raise ValueError('Backend selections must be unique.')
    if args.export:
        allowed = (ROOT/'experiments/results/llama_cpp').resolve()
        if not args.export.resolve().is_relative_to(allowed) or args.export.exists():
            raise ValueError('Public export must be a new directory under experiments/results/llama_cpp/.')
    for path in texts:
        if not path.read_text(encoding='utf-8').strip():
            raise ValueError(f'Empty passage: {path.name}')
    if any(name.startswith('python') for name in args.backends):
        if not args.reference or not args.reference.is_file():
            raise ValueError('Python comparisons require consented local_data/reference.wav. Use --backends llama for a synthesis-only measurement.')
        if 'python-transcript' in args.backends and (not args.transcript or not args.transcript.is_file() or not args.transcript.read_text().strip()):
            raise ValueError('Transcript conditioning requires the exact local reference transcript.')
    base = RunConfig(text='preflight', reference=args.reference, frames=args.frames, timeout=args.timeout)
    # Enforce the same executable/model/reference and private-output rules for the whole run.
    if 'llama' in args.backends:
        validate(base)
    elif args.reference and not args.reference.resolve().is_relative_to((ROOT/'local_data').resolve()):
        raise ValueError('Keep reference audio inside ignored local_data/.')
    if args.transcript and not args.transcript.resolve().is_relative_to((ROOT / 'local_data').resolve()):
        raise ValueError('Reference transcript must remain in ignored local_data/.')
    root = ROOT / 'outputs/benchmark'
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='batch-', dir=root))
    rows = []
    for backend in args.backends:
        if backend == 'llama':
            for passage in texts:
                text = passage.read_text(encoding='utf-8').strip()
                for repetition in range(args.runs):
                    try:
                        result_dir = execute(RunConfig(text=text, reference=args.reference, frames=args.frames,
                                                      timeout=args.timeout, seed=args.seed+repetition, outputs=directory/'llama'))
                    except RunError as exc:
                        result_dir = exc.result_dir
                    row = json.loads((result_dir/'result.json').read_text())
                    row.update(passage=passage.stem, run=repetition+1, timing_scope='fresh CLI process; loading included only in process RTF')
                    rows.append(row)
                    (directory/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
                    print(f'{backend} {passage.stem} run {repetition+1}: {row["status"]}', flush=True)
        else:
            mode = 'speaker' if backend == 'python-speaker' else 'transcript'
            worker = directory / backend
            worker.mkdir()
            revision = (ROOT/'config/python_model_revision.txt').read_text().strip()
            command = [sys.executable, '-m', 'qwen3_tts_lab.python_benchmark_worker', '--mode', mode,
                       '--reference', str(args.reference.resolve()), '--texts', str(args.texts.resolve()),
                       '--destination', str(worker.resolve()), '--runs', str(args.runs), '--frames', str(args.frames),
                       '--seed', str(args.seed), '--revision', revision]
            if args.transcript:
                command.extend(['--transcript', str(args.transcript.resolve())])
            with (worker/'process.log').open('w') as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.timeout*len(texts)*args.runs, check=False)
                    failure = result.returncode != 0
                except subprocess.TimeoutExpired:
                    failure = True
            if (worker/'results.json').is_file():
                rows.extend(json.loads((worker/'results.json').read_text()))
            if failure:
                # Preserve a setup/process failure even when no synthesis row could be written.
                rows.append({'backend': 'python', 'conditioning': 'speaker_only' if mode == 'speaker' else 'audio_and_transcript',
                             'passage': 'worker_setup_or_process', 'run': 0, 'status': 'error', 'listening_assessment': 'pending'})
            (directory/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    (directory/'summary.json').write_text(json.dumps(summarize(rows), indent=2)+'\n')
    if args.export:
        allowed = (ROOT/'experiments/results/llama_cpp').resolve()
        if not args.export.resolve().is_relative_to(allowed):
            raise ValueError('Public export must be under experiments/results/llama_cpp/ in a new directory.')
        export_results(rows, args.export)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends', nargs='+', choices=('llama','python-speaker','python-transcript'), default=['llama','python-speaker','python-transcript'])
    parser.add_argument('--texts', type=Path, default=ROOT/'experiments/texts')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--transcript', type=Path)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--frames', type=int, default=1200)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--export', type=Path, help='New directory under experiments/results/llama_cpp for sanitized measurement export.')
    args = parser.parse_args()
    try:
        directory = benchmark(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(directory)
    rows = json.loads((directory/'results.json').read_text())
    if any(row['status'] != 'ok' for row in rows):
        parser.exit(1, 'One or more runs failed; inspect the local logs and exported failure counts.\n')


if __name__ == '__main__':
    main()
