#!/usr/bin/env bash
#SBATCH --job-name=build_bp_v2_100k
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

OUT_ROOT="${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_tiny_100kexamples_20260713"
TMP_ROOT="${OUT_ROOT}.tmp"
HF_REPO_ID="${HF_REPO_ID:-flaitenberger/BranchProof-unique-v2}"
SUBSET="train_fixedtarget_up_to_10_100k"

rm -rf "${TMP_ROOT}"
python -m synthrlvl.datasets.materialize \
  --output-root "${TMP_ROOT}" \
  --train-up-to-3-subset unused_up_to_3 \
  --train-up-to-5-subset unused_up_to_5 \
  --train-up-to-10-subset "${SUBSET}" \
  --train-up-to-15-subset unused_up_to_15 \
  --train-up-to-20-subset unused_up_to_20 \
  --train-up-to-25-subset unused_up_to_25 \
  --train-up-to-3-rows 0 \
  --train-up-to-5-rows 0 \
  --train-up-to-10-rows 101000 \
  --train-up-to-15-rows 0 \
  --train-up-to-20-rows 0 \
  --train-up-to-25-rows 0 \
  --val-rows-per-step 0 \
  --val-max-step 0 \
  --seed 3407 \
  --distractor-ratio 0.5 \
  --difficulty hard_fsa_schema \
  --branching-factor 4 \
  --chunk-size 10000 \
  --require-unique-sequences

OUT_ROOT_CHECK="${TMP_ROOT}" SUBSET_CHECK="${SUBSET}" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

path = Path(os.environ["OUT_ROOT_CHECK"]) / os.environ["SUBSET_CHECK"] / "train.parquet"
table = pq.read_table(path)
assert table.num_rows == 101_000

depth_counts = {depth: 0 for depth in range(1, 11)}
record_keys = set()
logic_hashes = set()
nl_hashes = set()
selected_indices = []
logic_fields = (
    "constants",
    "predicates",
    "premises_fol",
    "proof_fol",
    "question_fol",
    "answer",
)
nl_fields = ("premises_nl", "proof_nl", "question_nl", "answer")

for index, row in enumerate(table.to_pylist()):
    depth = int(row["depth"])
    if depth_counts[depth] >= 10_000:
        continue
    logic_payload = {key: row[key] for key in logic_fields}
    nl_payload = {key: row[key] for key in nl_fields}
    logic_hash = hashlib.sha256(
        json.dumps(logic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    nl_hash = hashlib.sha256(
        json.dumps(nl_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    if logic_hash in logic_hashes or nl_hash in nl_hashes:
        continue
    selected_indices.append(index)
    logic_hashes.add(logic_hash)
    nl_hashes.add(nl_hash)
    depth_counts[depth] += 1
    record_keys.add((depth, int(row["record_index"])))

assert depth_counts == {depth: 10_000 for depth in range(1, 11)}, depth_counts
assert len(record_keys) == 100_000
assert len(logic_hashes) == 100_000
assert len(nl_hashes) == 100_000
selected = table.take(pa.array(selected_indices, type=pa.int64()))
output = path.with_suffix(".unique.parquet")
pq.write_table(selected, output)
output.replace(path)
print(
    {
        "rows": selected.num_rows,
        "depth_counts": depth_counts,
        "unique_logic_sequences": len(logic_hashes),
        "unique_nl_sequences": len(nl_hashes),
    }
)
PY

rm -rf "${OUT_ROOT}"
mv "${TMP_ROOT}" "${OUT_ROOT}"

OUT_ROOT_PUSH="${OUT_ROOT}" HF_REPO_ID_PUSH="${HF_REPO_ID}" SUBSET_PUSH="${SUBSET}" python - <<'PY'
import os

from datasets import load_dataset
from synthetic_dataset import MaterializedDatasetBuilder, MaterializedSyntheticDataset

subset = os.environ["SUBSET_PUSH"]
dataset = MaterializedSyntheticDataset(train_up_to_10_subset=subset)
MaterializedDatasetBuilder(dataset).push_to_hub(
    output_root=os.environ["OUT_ROOT_PUSH"],
    repo_id=os.environ["HF_REPO_ID_PUSH"],
    private=True,
)
remote = load_dataset(os.environ["HF_REPO_ID_PUSH"], subset, split="train")
assert len(remote) == 100_000
print({"remote_subset": subset, "remote_rows": len(remote)})
PY
