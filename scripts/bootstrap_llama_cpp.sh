#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
./scripts/doctor.sh
revision="$(cat config/llama_cpp_revision.txt)"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full pinned llama.cpp SHA.' >&2; exit 1; }
jobs="${LLAMA_BUILD_JOBS:-8}"
[[ "$jobs" =~ ^[0-9]+$ ]] && (( 10#$jobs >= 1 && 10#$jobs <= 32 )) || { echo 'LLAMA_BUILD_JOBS must be 1..32.' >&2; exit 1; }
export PATH="$project_root/.tools/cmake-venv/bin:$project_root/.tools/cuda-12.8.1/bin:$PATH"
for program in git cmake c++ nvcc; do
  command -v "$program" >/dev/null || { echo "Missing $program. Run scripts/setup_llama_toolchain.sh or provide a compatible toolchain on PATH." >&2; exit 1; }
done
checkout="$project_root/.tools/llama.cpp"
remote='https://github.com/ggml-org/llama.cpp.git'
if [[ ! -e "$checkout" ]]; then
  git clone --no-checkout "$remote" "$checkout"
fi
[[ -d "$checkout/.git" ]] || { echo 'Unexpected .tools/llama.cpp; expected a Git checkout.' >&2; exit 1; }
[[ "$(git -C "$checkout" remote get-url origin)" == "$remote" ]] || { echo 'Unexpected llama.cpp remote; inspect checkout before retrying.' >&2; exit 1; }
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]] || { echo 'Dirty llama.cpp checkout; preserve local changes before retrying.' >&2; exit 1; }
if ! git -C "$checkout" cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C "$checkout" fetch --depth 1 origin "$revision"
fi
git -C "$checkout" checkout --detach "$revision"
cuda_root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
cmake -S "$checkout" -B "$checkout/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDAToolkit_ROOT="$cuda_root" -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$checkout/build" --config Release --target llama-tts -j "$jobs"
"$checkout/build/bin/llama-tts" --help
