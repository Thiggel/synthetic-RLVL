#!/usr/bin/env bash
# L1 convergence study (2026-09-30): held-out evals of the kept checkpoints (every 250 steps) of both arms,
# the Dolci gate (950 items) and the in-domain generator test (2000 items), greedy, frozen pre-libext checker.
# Also G13 (2026-10-01: GRPO from the libext le SFT arm), with the new-library checker snapshot.
# Submits one grpo_gate_ckpts job per arm for the kept checkpoints missing either eval, unless one is queued.
# Idempotent: rerun from each loop tick. DEP=<sbatch dependency> holds the job (the L40 cap on gruenau12 is 5).
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
ROOT=/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928
OLD=/vol/tmp2/laitenbf/rlvl_data/checker_snapshot_pre_libext_20260930
NEW=/vol/tmp2/laitenbf/rlvl_data/checker_snapshot_libext_20261001
for spec in "L1_correct:${OLD}" "L1_cvf:${OLD}" "2b_le_G13_cvffmt:${NEW}"; do
  arm=${spec%%:*} S=${spec#*:}
  if squeue -u laitenbf -h -n "gate_${arm}" | grep -q .; then echo "${arm}: gate job queued"; continue; fi
  todo=()
  for c in "${ROOT}/${arm}"/checkpoint-* "${ROOT}/${arm}/final"; do
    [[ -d ${c} ]] || continue
    n=$(basename "${c}")
    [[ ${n} == final ]] || (( ${n#checkpoint-} % 250 == 0 )) || continue
    [[ -s ${c}/rl_gate_dolci/summary.json && -s ${c}/formal_eval/summary.json ]] && continue
    ls "${c}"/model*.safetensors >/dev/null 2>&1 || continue
    todo+=("${arm}:${n}")
  done
  if (( ${#todo[@]} == 0 )); then echo "${arm}: nothing to evaluate"; continue; fi
  RLVL_PYTHONPATH="${S}/gen:${S}/rlvl_python" FORMAL_EVAL=1 sbatch --export=ALL --job-name="gate_${arm}" \
    -w gruenau12 ${DEP:+--dependency="${DEP}"} scripts/slurm/jobs/grpo_gate_ckpts.slurm "${todo[@]}"
  echo "${arm}: ${todo[*]}"
done
