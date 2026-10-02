#!/usr/bin/env bash
# L1 convergence study (2026-09-30): held-out evals of the kept checkpoints (every 250 steps) of both arms,
# the Dolci gate (950 items) and the in-domain generator test (2000 items), greedy, frozen pre-libext checker.
# Also G13 (2026-10-01: GRPO from the libext le SFT arm) and G14 (G13 + truncated completions in the loss with an
# overlong penalty), with the new-library checker snapshot. G15 (2026-10-02): the G14 recipe from the e2 SFT arm (EI
# round 2, teacher = an RL'd policy). G16: G15 + a no-proof penalty.
# Submits one grpo_gate_ckpts job per arm for the kept checkpoints missing either eval, unless one is queued.
# Idempotent: rerun from each loop tick. The L40 cap on gruenau12 is CAP=5 of my jobs: once my running jobs there
# plus pending gate jobs reach it, each new gate job waits (afterany) for the newest gate job. DEP=<dependency>
# overrides.
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
ROOT=/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928
OLD=/vol/tmp2/laitenbf/rlvl_data/checker_snapshot_pre_libext_20260930
NEW=/vol/tmp2/laitenbf/rlvl_data/checker_snapshot_libext_20261001
CAP=${CAP:-5}
used=$(( $(squeue -u laitenbf -h -t R -w gruenau12 | wc -l) + $(squeue -u laitenbf -h -t PD -o %j | grep -c '^gate_' || true) ))
last=$(squeue -u laitenbf -h -o '%i %j' | awk '$2 ~ /^gate_/ {print $1}' | sort -n | tail -1)
for spec in "L1_correct:${OLD}" "L1_cvf:${OLD}" "2b_le_G13_cvffmt:${NEW}" "2b_le_G14_cvffmt_overlong:${NEW}" \
            "2b_e2_G15_cvffmt_overlong:${NEW}" "2b_e2_G16_cvffmt_overlong_noproof:${NEW}"; do
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
  dep=${DEP:-}
  if [[ -z ${dep} ]] && (( used >= CAP )) && [[ -n ${last} ]]; then dep=afterany:${last}; fi
  last=$(RLVL_PYTHONPATH="${S}/gen:${S}/rlvl_python" FORMAL_EVAL=1 sbatch --parsable --export=ALL \
    --job-name="gate_${arm}" -w gruenau12 ${dep:+--dependency="${dep}"} scripts/slurm/jobs/grpo_gate_ckpts.slurm \
    "${todo[@]}")
  used=$((used + 1))
  echo "${arm}: ${todo[*]} -> job ${last}${dep:+ (after ${dep})}"
done
