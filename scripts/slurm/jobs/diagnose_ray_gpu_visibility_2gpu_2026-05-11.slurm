#!/usr/bin/env bash
# Minimal diagnostic for Ray GPU visibility under a 2-GPU Slurm allocation.
#SBATCH --job-name=diag_ray_gpu_2
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader

python - <<'PY'
import os
import ray
import torch

print("driver CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("driver torch device count", torch.cuda.device_count())
ray.init(num_cpus=8, include_dashboard=False)
print("ray cluster resources", ray.cluster_resources())

@ray.remote(num_gpus=1)
def probe(i):
    import os
    import ray
    import torch

    ids = ray.get_runtime_context().get_accelerator_ids()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    bus = None
    name = None
    if count:
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        bus = f"{props.pci_bus_id:08x}" if hasattr(props, "pci_bus_id") else "unknown"
        name = props.name
    return {
        "i": i,
        "accelerator_ids": ids,
        "CUDA_VISIBLE_DEVICES": visible,
        "torch_device_count": count,
        "device0_name": name,
        "device0_bus": bus,
    }

for result in ray.get([probe.remote(i) for i in range(2)]):
    print(result)
ray.shutdown()
PY
