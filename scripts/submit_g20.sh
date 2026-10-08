#!/usr/bin/env bash
# G20 (2026-10-08): the G19 recipe with the shaped reward c_cvf_fmt = correct x format_ok x (0.5 + 0.5 valid x prem_ok)
# instead of cvf_fmt, so a correct answer with an invalid proof earns 0.5 (user, 2026-10-08: "We could also try doing a
# correctness+validity reward?"). Every cvf run moved validity and none moved correctness (stage3 report, Result 16ag).
# G19 header follows:
# G19 (2026-10-05): the G18 recipe (cvf_fmt, gen + Dolci prompts, overlong + no-proof penalties, 1000 steps) from the
# best EI SFT arm, plus a KL anchor to the init (--beta, default 0.02). G16/G17 lost generator-test answer accuracy
# (.90 -> .76 greedy, `tools` valid -> .01) between steps 250 and 1000 although 3,000 of the 7,643 RL prompts are
# generator prompts: saturated gen groups carry no advantage, so nothing holds them in place.
# Usage: bash scripts/submit_g20.sh <ei arm, e.g. e6> [beta]. One H100 NVL on gruenau11.
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
ARM_EI=${1:?ei arm}; BETA=${2:-0.02}
# Memory (the KL reference model adds ~4.5 GB to G18's footprint): G18's --vllm-mem 0.2 OOMed at step 1 in the fp32
# logits by ~0.3 GB with 3.15 GiB reserved-but-unallocated (job 12165); 0.15 starves vLLM (262144-token max_model_len,
# job 12174; with a 16384 cap, Mamba cache blocks < max_num_seqs, job 12221). With sleep mode vLLM is offloaded during
# the training step anyway, so keep G18's 0.2 and let PyTorch use expandable segments against the fragmentation
# (vLLM's CuMemAllocator disables them inside its own pool). The recipe is unchanged.
VLLM_MEM=${VLLM_MEM:-0.2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=/vol/tmp2/laitenbf/rlvl_data
S=${DATA}/checker_snapshot_libext_20261001
export POLICY=qwen35_2b_lc_libext_${ARM_EI}_lr5em6_seed3407 ARM=c_cvf_fmt \
  RUN_NAME=2b_${ARM_EI}_G20_ccvffmt_overlong_noproof_kl${BETA#0.} RLVL_PYTHONPATH=${S}/gen:${S}/rlvl_python \
  MIN_FREE_MIB=37000 WAIT_FREE_MIN=1440
export EXTRA_ARGS="--resumable --benches gen,dolci_math,dolci_wordprob,dolci_yesno --max-per-bench 3000 --max-steps 1000 \
--save-steps 50 --keep-every 250 --per-device-batch 1 --vllm-mem ${VLLM_MEM} --beta ${BETA} \
--exclude-ids ${DATA}/datasets/rl_gate_dolci_instruct_20260928/contamination.json --no-mask-truncated \
--overlong-penalty 0.5 --no-proof-penalty 0.5"
j=$(sbatch --parsable --export=ALL -p gpu-staff --gres=gpu:h100nvl:1 -w gruenau11 --cpus-per-task=16 --mem=160G \
  --time=2-00:00:00 --job-name=grpo_G20 scripts/slurm/jobs/grpo_formal.slurm)
j2=$(sbatch --parsable --export=ALL -p gpu-staff --gres=gpu:h100nvl:1 -w gruenau11 --cpus-per-task=16 --mem=160G \
  --time=2-00:00:00 --job-name=grpo_G20 --dependency=singleton scripts/slurm/jobs/grpo_formal.slurm)
echo "G20 ${RUN_NAME}: ${j} -> ${j2}"
