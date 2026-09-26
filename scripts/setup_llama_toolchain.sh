#!/usr/bin/env bash
# Optional, project-local Linux x86_64 compiler toolkit; never installs a driver.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
./scripts/doctor.sh
[[ "$(uname -m)" == x86_64 ]] || { echo 'This pinned toolkit supports Linux x86_64 only.' >&2; exit 1; }
.venv/bin/python -m venv .tools/cmake-venv
.tools/cmake-venv/bin/python -m pip install 'cmake==3.31.6' 'ninja==1.11.1.4'
.venv/bin/python -m qwen3_tts_lab.artifacts config/cuda_toolchain.json .tools/cuda-archives
.venv/bin/python - <<'PY'
import shutil
import tarfile
from pathlib import Path
from qwen3_tts_lab.artifacts import read_manifest, matches
root = Path.cwd()
manifest = read_manifest(root / 'config/cuda_toolchain.json')
destination = root / '.tools/cuda-12.8.1'
marker = destination / '.manifest.json'
source = root / 'config/cuda_toolchain.json'
if destination.exists():
    if not marker.is_file() or marker.read_bytes() != source.read_bytes():
        raise SystemExit('Unexpected/incomplete CUDA directory; move it aside before retrying.')
    print('Reusing verified toolkit installation; run bootstrap to validate compiler usability.')
else:
    # Extract into a staging location so interruption cannot masquerade as a complete toolkit.
    staging = root / '.tools/cuda-12.8.1.staging'
    staging.mkdir(exist_ok=False)
    for item in manifest['files']:
        archive = root / '.tools/cuda-archives' / item['filename']
        if not matches(archive, item):
            raise SystemExit(f'Invalid archive: {archive.name}')
        with tarfile.open(archive) as tar:
            tar.extractall(staging / 'components', filter='data')
        component = staging / 'components' / item['filename'].removesuffix('.tar.xz')
        shutil.copytree(component, staging / 'toolkit', dirs_exist_ok=True, symlinks=True)
    shutil.copyfile(source, staging / 'toolkit/.manifest.json')
    (staging / 'toolkit').rename(destination)
    print('Installed project-local CUDA toolkit; component licenses retained in staging/components.')
# NVIDIA component archives use lib/, while nvcc's profile expects lib64/.
link = destination / 'lib64'
if not link.exists() and not link.is_symlink():
    link.symlink_to('lib', target_is_directory=True)
PY
.tools/cuda-12.8.1/bin/nvcc --version
