#!/usr/bin/env bash
# One-time setup of the Nebius H100 VM. Run as a sudo-capable user:
#   cp infra/env.example infra/.env && $EDITOR infra/.env
#   bash infra/setup_vm.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/.env"
H=/opt/hackathon

echo "== system packages"
sudo apt-get update -y
sudo apt-get install -y git build-essential tmux curl jq ffmpeg

echo "== GPU / CUDA toolkit"
[[ -d /usr/local/cuda/bin ]] && export PATH=/usr/local/cuda/bin:$PATH
nvidia-smi
if ! command -v nvcc >/dev/null; then
  echo "!! nvcc not found. Marlin (JIT) and CUDA-C++ agents need the CUDA toolkit (12.4+)."
  echo "!! Use the ubuntu24.04-cuda12 Nebius image (infra/terraform does), or install cuda-toolkit-12-8, then re-run."
  exit 1
fi
nvcc --version | tail -1
sudo nvidia-smi -pm 1
# Stable clocks make numbers comparable across runs. 1980 MHz = H100 SXM max boost.
if [[ "${LOCK_CLOCKS:-1}" == "1" ]]; then sudo nvidia-smi -lgc 1980,1980 || true; fi

echo "== layout"
sudo mkdir -p "$H"/{shared,workers}
sudo chown -R "$USER" "$H"
[[ -d "$H/main" ]] || git clone "$REPO_URL" "$H/main"

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== main venv (sglang + DSLs). sglang pins torch, so install it first."
uv venv --python 3.12 "$H/venv"
uv pip install --python "$H/venv/bin/python" sglang
uv pip install --python "$H/venv/bin/python" fastapi uvicorn numpy ninja huggingface_hub \
  tilelang nvidia-cutlass-dsl
"$H/venv/bin/python" -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name())"

echo "== vLLM venv (Machete baseline only)"
uv venv --python 3.12 "$H/venv-vllm"
uv pip install --python "$H/venv-vllm/bin/python" vllm fastapi uvicorn numpy

echo "== CUTLASS headers (for CUDA C++ agents)"
[[ -d "$H/cutlass" ]] || git clone --depth 1 https://github.com/NVIDIA/cutlass "$H/cutlass"

echo "== Devin CLI"
command -v devin >/dev/null || curl -fsSL https://cli.devin.ai/install.sh | bash

echo "== smoke test"
cd "$H/main"
"$H/venv/bin/python" -m harness.check --baseline torch_dequant --quick
"$H/venv/bin/python" -m harness.check --kernel kernels/_template --quick
echo
echo "Done. Next: bash infra/start_board.sh && bash infra/start_workers.sh 5"
