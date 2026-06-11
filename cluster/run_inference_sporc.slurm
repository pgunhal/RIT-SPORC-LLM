#!/bin/bash

# CHANGE HERE:
# - Set your RC account.
# - Set the partition that should run the job.
# - Set requested wall time / CPU / memory / GPU.
# - Use gpu:1 for flexible placement, or gpu:a100:1 if you require that hardware.
#SBATCH --job-name=sporc-infer
#SBATCH --account=reu-aisocial
#SBATCH --partition=sporc-gpu
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# CHANGE HERE:
# - PROJECT_ROOT should match the repo path on SPORC.
# - SPACK_ENV_NAME can be replaced if your cluster environment changes.
# - MODEL_NAME_OR_PATH can be any Hugging Face causal LM that fits the requested GPU.
# - INPUT_PATH and OUTPUT_PATH should point at your task file and desired output location.
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/SPORC-LLM-Inference}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-default-ml-x86_64-25052701}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data/sample_inputs.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/runs/inference_${SLURM_JOB_ID:-manual}.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HF_TORCH_DTYPE="${HF_TORCH_DTYPE:-auto}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-auto}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/runs" "${HF_HOME}"
cd "${PROJECT_ROOT}"

if [ -f /tools/spack/share/spack/setup-env.sh ]; then
  source /tools/spack/share/spack/setup-env.sh
elif [ -f /.autofs/tools/spack/share/spack/setup-env.sh ]; then
  source /.autofs/tools/spack/share/spack/setup-env.sh
else
  echo "Could not find Spack setup-env.sh" >&2
  exit 2
fi

spack env activate "${SPACK_ENV_NAME}"
PYTHON_CMD="$(command -v python3)"

export HF_HOME

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "hostname=$(hostname)"
echo "python_cmd=${PYTHON_CMD}"
nvidia-smi || true

"${PYTHON_CMD}" - <<'PY'
import sys
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.stderr.write("No CUDA GPU visible to PyTorch in this allocation.\n")
    sys.exit(2)
PY

"${PYTHON_CMD}" -u llm/infer_hf.py \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --hf-device-map "${HF_DEVICE_MAP}" \
  --hf-torch-dtype "${HF_TORCH_DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
