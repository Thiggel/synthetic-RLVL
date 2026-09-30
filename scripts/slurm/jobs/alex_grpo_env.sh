# Sourced by alex_grpo_formal.slurm and alex_grpo_gate_ckpts.slurm (alex, rtxpro6k).
# Unpacks the Stage-2 GRPO env to the node-local $TMPDIR and points every cache there:
# nothing is extracted on atuin/vault (500K-file quota) and nothing is written to /home/hpc.
# The env is gruenau's /vol/tmp2/laitenbf/.venv_rlvl_grpo (TRL 1.14, vLLM 0.30, torch 2.13+cu130,
# transformers 5.17) plus its uv base python 3.12.12, packed verbatim (not re-resolved) on 2026-09-30:
#   tar -C / --transform 's|^vol/home-vol2/ml/laitenbf/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu|rlvl_grpo_env/python|;s|^vol/tmp2/laitenbf/.venv_rlvl_grpo|rlvl_grpo_env/venv|' \
#     -cf - vol/home-vol2/ml/laitenbf/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu vol/tmp2/laitenbf/.venv_rlvl_grpo \
#     | zstd -T16 -6 -o rlvl_grpo_env_20260930.tar.zst       # then rsync to ${ALEX_VAULT}/rlvl_envs/
# Data (RLVL_DATA_ROOT) mirrors gruenau's /vol/tmp2/laitenbf/rlvl_data; hf_home/ holds the offline
# allenai/Dolci-Instruct-RL hub + datasets caches. The checker is a frozen snapshot of RLVL-next
# (gen/rlvlgen + rlvl/python with _rlvl.abi3.so = _rlvl.abi3.so.pre_libext_20260930), so rewards stay
# constant while the lemma library is extended on gruenau.
ALEX_WORK=/home/atuin/c107fa/c107fa12
ALEX_VAULT=/home/vault/c107fa/c107fa12
export RLVL_DATA_ROOT="${RLVL_DATA_ROOT:-${ALEX_WORK}/rlvl_data}"
ENV_TAR="${ENV_TAR:-${ALEX_VAULT}/rlvl_envs/rlvl_grpo_env_20260930.tar.zst}"
CHECKER="${CHECKER:-${RLVL_DATA_ROOT}/RLVL-next_snapshot_20260930}"
if [[ -z "${TMPDIR:-}" ]]; then  # Slurm on alex sets a job-specific TMPDIR (removed at job end); else own one
  export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
  trap 'rm -rf "${TMPDIR}"' EXIT
fi
mkdir -p "${TMPDIR}"
ENV_DIR="${TMPDIR}/rlvl_grpo_env"
if [[ ! -x "${ENV_DIR}/venv/bin/python" ]]; then
  t0=$(date +%s)
  zstd -dc -T0 "${ENV_TAR}" | tar -C "${TMPDIR}" -xf -
  # relocate: venv -> unpacked base python, console-script shebangs -> unpacked venv
  sed -i "s|^home = .*|home = ${ENV_DIR}/python/bin|" "${ENV_DIR}/venv/pyvenv.cfg"
  ln -sfn "${ENV_DIR}/python/bin/python3.12" "${ENV_DIR}/venv/bin/python"
  grep -lI '^#!/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python' "${ENV_DIR}"/venv/bin/* \
    | xargs -r sed -i "1s|^#!/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python|#!${ENV_DIR}/venv/bin/python|"
  echo "unpacked ${ENV_TAR} to ${ENV_DIR} in $(( $(date +%s) - t0 )) s"
fi
VENV="${ENV_DIR}/venv"
export PATH="${VENV}/bin:${PATH}" VIRTUAL_ENV="${VENV}"
export HF_HOME="${RLVL_DATA_ROOT}/hf_home" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE="${TMPDIR}/hf_modules"
# same CUDA toolkit as gruenau (cuda-13.2), from the alex module tree (module cuda/13.2.0)
export CUDA_HOME="${CUDA_HOME:-/apps/SPACK/1.1.1-alex/opt/spack/linux-almalinux9-zen3/none-none/cuda-13.2.0-muu4y5j66hnxz4ijk3xkmchxguoddbsh}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# every JIT/compile cache node-local ($HOME is /home/hpc)
C="${TMPDIR}/cache"
export XDG_CACHE_HOME="${C}" TRITON_HOME="${C}" TRITON_CACHE_DIR="${C}/triton" VLLM_CACHE_ROOT="${C}/vllm" \
  VLLM_CONFIG_ROOT="${C}/vllm_config" TORCH_EXTENSIONS_DIR="${C}/torch_extensions" TORCHINDUCTOR_CACHE_DIR="${C}/inductor" \
  FLASHINFER_WORKSPACE_BASE="${C}" CUDA_CACHE_PATH="${C}/nv" MPLCONFIGDIR="${C}/mpl" PYTHONPYCACHEPREFIX="${C}/pycache" \
  UV_CACHE_DIR="${C}/uv" PIP_CACHE_DIR="${C}/pip"
export PYTHONPATH="${CHECKER}/gen:${CHECKER}/rlvl/python"
hash -r 2>/dev/null || true
