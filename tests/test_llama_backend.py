from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qwen3_tts_lab import artifacts, llama_runner
from qwen3_tts_lab.metrics import real_time_factor, summarize


def manifest_item(data=b'weights'):
    return {'filename': 'tiny.gguf', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'url': 'https://example.org/tiny.gguf'}


def test_manifest_rejects_traversal_duplicate_and_bad_integrity(tmp_path):
    path = tmp_path / 'manifest.json'
    for changes in ({'filename': '../escape'}, {'sha256': 'bad'}, {'size': 0}, {'url': 'http://example.org/a'}):
        path.write_text(json.dumps({'files': [manifest_item() | changes]}))
        with pytest.raises(ValueError):
            artifacts.read_manifest(path)
    path.write_text(json.dumps({'files': [manifest_item(), manifest_item()]}))
    with pytest.raises(ValueError):
        artifacts.read_manifest(path)


def test_retrieve_reuses_verified_file_and_preserves_corrupt_target(tmp_path, monkeypatch):
    path = tmp_path / 'tiny.gguf'
    path.write_bytes(b'weights')
    monkeypatch.setattr(artifacts.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not download valid existing artifact'))
    artifacts.retrieve({'files': [manifest_item()]}, tmp_path)
    path.write_bytes(b'private')
    with pytest.raises(ValueError, match='Unexpected artifact'):
        artifacts.retrieve({'files': [manifest_item()]}, tmp_path)
    assert path.read_bytes() == b'private'


def test_retrieve_validates_download_before_publication(tmp_path, monkeypatch):
    def download(args, **kwargs):
        Path(args[args.index('--output') + 1]).write_bytes(b'wrong')
    monkeypatch.setattr(artifacts.subprocess, 'run', download)
    with pytest.raises(ValueError, match='Checksum/size mismatch'):
        artifacts.retrieve({'files': [manifest_item()]}, tmp_path)
    assert not (tmp_path / 'tiny.gguf').exists()
    def correct(args, **kwargs):
        Path(args[args.index('--output') + 1]).write_bytes(b'weights')
    monkeypatch.setattr(artifacts.subprocess, 'run', correct)
    artifacts.retrieve({'files': [manifest_item()]}, tmp_path)
    assert (tmp_path / 'tiny.gguf').read_bytes() == b'weights'
    assert not (tmp_path / 'tiny.gguf.partial').exists()


def test_retrieve_refuses_symlinks(tmp_path):
    private = tmp_path / 'private'
    private.write_bytes(b'weights')
    (tmp_path / 'tiny.gguf').symlink_to(private)
    with pytest.raises(ValueError):
        artifacts.retrieve({'files': [manifest_item()]}, tmp_path)
    assert private.read_bytes() == b'weights'


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_runner, 'ROOT', tmp_path)
    folder = tmp_path / 'paths with spaces'
    folder.mkdir()
    exe = folder / 'fake tts'
    exe.write_text(f'''#!{sys.executable}
import json,sys,wave,struct
from pathlib import Path
Path('args.json').write_text(json.dumps(sys.argv))
out=Path(sys.argv[sys.argv.index('--output')+1])
with wave.open(str(out),'wb') as f:
 f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(struct.pack('<h',1000)*24000)
print('generated 12 frames, 48044 bytes of WAV audio (24000 Hz)')
print('timings: prompt eval 0.10s + generation 0.20s + vocoder 0.05s = total 0.35s')
'''.replace("Path('args.json')", "Path(sys.argv[sys.argv.index('--output')+1]).with_name('args.json')"))
    exe.chmod(0o755)
    model, companion, reference = [folder / name for name in ('talker.gguf', 'audio.gguf', 'reference.wav')]
    for path in (model, companion, reference):
        path.write_bytes(b'keep')
    class Monitor:
        def __init__(self, *args): pass
        def start(self): pass
        def finish(self): return {}
    monkeypatch.setattr(llama_runner, 'ResourceMonitor', Monitor)
    return llama_runner.RunConfig(text='Grüße "$(touch nope)"; Straße', executable=exe, model=model,
                                  mmproj=companion, reference=reference, outputs=tmp_path / 'outputs/llama-cpp')


def test_runner_arguments_unicode_paths_and_unique_outputs(config):
    first = llama_runner.execute(config)
    second = llama_runner.execute(config)
    assert first != second
    args = json.loads((first / 'args.json').read_text())
    assert args[args.index('-p') + 1] == config.text
    assert args[args.index('--tts-speaker-file') + 1] == str(config.reference.resolve())
    assert config.reference.read_bytes() == b'keep'
    record = json.loads((first / 'result.json').read_text())
    assert record['status'] == 'ok'
    assert record['sample_rate'] == 24000
    assert record['synthesis_rtf'] == pytest.approx(0.35)
    assert record['listening_assessment'] == 'pending'


@pytest.mark.parametrize('changes', [{'frames': 0}, {'timeout': float('nan')}, {'top_p': 2}, {'temperature': float('inf')}, {'seed': -1}, {'language': 'xx'}])
def test_runner_rejects_invalid_settings(config, changes):
    with pytest.raises(ValueError):
        llama_runner.execute(replace(config, **changes))
    assert not config.outputs.exists()


def test_runner_missing_files_and_output_boundary(config):
    with pytest.raises(ValueError, match='missing'):
        llama_runner.execute(replace(config, model=config.model.with_name('absent')))
    with pytest.raises(ValueError, match='ignored outputs'):
        llama_runner.execute(replace(config, outputs=config.reference.parent))


def test_runner_preserves_failure_exit_code_and_logs(config):
    config.executable.write_text('#!/bin/sh\necho known-failure >&2\nexit 7\n')
    with pytest.raises(llama_runner.RunError) as exc:
        llama_runner.execute(config)
    assert exc.value.returncode == 7
    assert 'known-failure' in (exc.value.result_dir / 'process.log').read_text()
    assert json.loads((exc.value.result_dir / 'result.json').read_text())['status'] == 'error'


def test_runner_timeout(config):
    config.executable.write_text('#!/bin/sh\nsleep 60\n')
    with pytest.raises(llama_runner.RunError) as exc:
        llama_runner.execute(replace(config, timeout=0.1))
    assert exc.value.returncode == 124
    assert (exc.value.result_dir / 'result.json').is_file()


def test_runner_rejects_frame_cap(config):
    with pytest.raises(llama_runner.RunError, match='frame limit'):
        llama_runner.execute(replace(config, frames=12))


def test_rtf_and_summary_preserve_failures():
    assert real_time_factor(2, 4) == 0.5
    for seconds, duration in [(1, 0), (-1, 4), (float('nan'), 4)]:
        with pytest.raises(ValueError): real_time_factor(seconds, duration)
    base = {'backend': 'llama.cpp', 'conditioning': 'none', 'passage': 'de_prose', 'status': 'ok'}
    result = summarize([base | {'process_rtf': 1}, base | {'process_rtf': 3}, base | {'status': 'error', 'process_rtf': 999}])[0]
    assert result['failures'] == 1
    assert result['process_rtf'] == {'median': 2, 'min': 1, 'max': 3}


def test_bootstrap_rerun_and_dirty_remote_guards(tmp_path):
    root = tmp_path / 'lab'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    scripts.joinpath('doctor.sh').write_text('#!/bin/sh\nexit 0\n')
    scripts.joinpath('doctor.sh').chmod(0o755)
    source = Path(__file__).parents[1] / 'scripts/bootstrap_llama_cpp.sh'
    scripts.joinpath(source.name).write_bytes(source.read_bytes())
    repo = root / '.tools/llama.cpp'
    repo.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init')
    git('config', 'user.email', 'test@example.org')
    git('config', 'user.name', 'Test')
    (repo / '.gitignore').write_text('build/\n')
    git('add', '.gitignore'); git('commit', '-m', 'fixture')
    git('remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git')
    config_dir = root / 'config'
    config_dir.mkdir()
    (config_dir / 'llama_cpp_revision.txt').write_text(git('rev-parse', 'HEAD'))
    tools = root / '.tools/cmake-venv/bin'
    tools.mkdir(parents=True)
    for name, body in [('nvcc', '#!/bin/sh\nexit 0\n'), ('cmake', '''#!/bin/sh
if [ "$1" = "--build" ]; then
 mkdir -p "$2/bin"
 printf '#!/bin/sh\\nexit 0\\n' > "$2/bin/llama-tts"
 chmod +x "$2/bin/llama-tts"
fi
''')]:
        path = tools / name; path.write_text(body); path.chmod(0o755)
    def run(): return subprocess.run(['bash', str(scripts / source.name)], capture_output=True, text=True)
    assert run().returncode == 0
    assert run().returncode == 0
    (repo / 'private-edit').write_text('keep')
    failed = run()
    assert failed.returncode == 1 and 'Dirty' in failed.stderr
    assert (repo / 'private-edit').read_text() == 'keep'
    (repo / 'private-edit').unlink()
    git('remote', 'set-url', 'origin', 'https://example.org/other.git')
    failed = run()
    assert failed.returncode == 1 and 'Unexpected llama.cpp remote' in failed.stderr
