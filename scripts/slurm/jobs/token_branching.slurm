#!/usr/bin/env bash
#SBATCH --job-name=tok_branch
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Token-level branching under the rlvl.Guide mask (scripts/analysis/token_branching.py).
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
ROOT=/vol/tmp2/laitenbf
export PATH="${ROOT}/.venv_rlvl_grpo/bin:${PATH}" HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=${ROOT}/rlvl_data/checker_snapshot_libext_20261001
export PYTHONPATH="$S/gen:$S/rlvl_python"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python scripts/analysis/token_branching.py ${ARGS:-}
