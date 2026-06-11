#!/bin/bash

# CHANGE HERE:
# - Set account / partition / runtime for the sweep workload.
# - Sweeps often need longer runtime than a single-run inference job.
#SBATCH --job-name=sporc-sweep
#SBATCH --account=reu-aisocial
#SBATCH --partition=sporc-gpu
#SBATCH --time=0-00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# CHANGE HERE:
# - FEW_SHOT_CONFIG should point at your demonstration pool.
# - INPUT_PATH should point at your task dataset.
# - OUTPUT_DIR controls where sweep outputs are written.
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/SPORC-LLM-Inference-Template}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-default-ml-x86_64-25052701}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data/sample_inputs.jsonl}"
FEW_SHOT_CONFIG="${FEW_SHOT_CONFIG:-${PROJECT_ROOT}/configs/few_shot_pool.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/prompt_sweep_${SLURM_JOB_ID:-manual}}"
HF_TORCH_DTYPE="${HF_TORCH_DTYPE:-auto}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-auto}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "${HF_HOME}"
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

"${PYTHON_CMD}" -u llm/run_prompt_experiments.py \
  --input "${INPUT_PATH}" \
  --few-shot-config "${FEW_SHOT_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --hf-device-map "${HF_DEVICE_MAP}" \
  --hf-torch-dtype "${HF_TORCH_DTYPE}"
