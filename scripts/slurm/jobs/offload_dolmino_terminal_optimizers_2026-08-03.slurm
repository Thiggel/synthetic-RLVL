#!/usr/bin/env bash
#SBATCH --job-name=offload_dolm_opt
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VAULT_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_dolmino_5b"
WORK_ROOT="${WORK}/synthetic-RLVL/checkpoint_state_offload/nanotron_dolmino_5b"
AUDIT_ROOT="${HPCVAULT}/synthetic-RLVL/analysis/dolmino_optimizer_offload_20260803"
mkdir -p "${WORK_ROOT}" "${AUDIT_ROOT}"

offload_optimizer() {
  local run_name="$1"
  local source="${VAULT_ROOT}/${run_name}/checkpoints/9537/optimizer"
  local destination="${WORK_ROOT}/${run_name}/checkpoints/9537/optimizer"
  local audit="${AUDIT_ROOT}/${run_name}.txt"
  local source_manifest destination_manifest

  if [[ -L "${source}" ]]; then
    test "$(readlink -f "${source}")" = "$(readlink -f "${destination}")"
    test "$(find "${destination}" -maxdepth 1 -type f -name 'optimizer_*.pt' | wc -l)" -eq 4
    printf 'already_offloaded\nsource=%s\ndestination=%s\n' \
      "${source}" "${destination}" > "${audit}"
    return
  fi

  test -d "${source}"
  test "$(find "${source}" -maxdepth 1 -type f -name 'optimizer_*.pt' | wc -l)" -eq 4
  mkdir -p "${destination}"
  source_manifest="$(mktemp)"
  destination_manifest="$(mktemp)"

  find "${source}" -maxdepth 1 -type f -name 'optimizer_*.pt' \
    -printf '%f %s\n' | sort > "${source_manifest}"
  awk '$2 < 20000000000 { exit 1 }' "${source_manifest}"

  rsync --archive --partial --remove-source-files \
    "${source}/" "${destination}/"

  find "${destination}" -maxdepth 1 -type f -name 'optimizer_*.pt' \
    -printf '%f %s\n' | sort > "${destination_manifest}"
  cmp "${source_manifest}" "${destination_manifest}"
  test "$(find "${source}" -type f | wc -l)" -eq 0
  find "${source}" -depth -delete
  ln -s "${destination}" "${source}"

  {
    printf 'accepted\nsource=%s\ndestination=%s\n' "${source}" "${destination}"
    cat "${destination_manifest}"
    printf 'total_bytes=%s\n' "$(awk '{ total += $2 } END { print total }' "${destination_manifest}")"
  } > "${audit}"
  find "${source_manifest}" "${destination_manifest}" -maxdepth 0 -type f -delete
}

offload_optimizer qwen25_7b_dolmino_control_p5_5b
offload_optimizer qwen25_7b_dolmino_logic_p5_5b

python scripts/nanotron/verify_training_checkpoint.py \
  --checkpoint "${VAULT_ROOT}/qwen25_7b_dolmino_control_p5_5b/checkpoints/9537" \
  --expected-step 9537 --sequence-length 4096 --global-batch-samples 128 \
  --tp 4 --dp 2 --model-files 625 --optimizer-shards 4 \
  --lr-scheduler-shards 4 --rng-shards 8 \
  --output "${AUDIT_ROOT}/control_checkpoint_after_offload.json"

python scripts/nanotron/verify_training_checkpoint.py \
  --checkpoint "${VAULT_ROOT}/qwen25_7b_dolmino_logic_p5_5b/checkpoints/9537" \
  --expected-step 9537 --sequence-length 4096 --global-batch-samples 128 \
  --tp 4 --dp 2 --model-files 625 --optimizer-shards 4 \
  --lr-scheduler-shards 4 --rng-shards 8 \
  --output "${AUDIT_ROOT}/logic_checkpoint_after_offload.json"
