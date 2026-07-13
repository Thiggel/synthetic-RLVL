from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_qwen_consumers_enforce_rope_preflight() -> None:
    for relative in (
        "scripts/slurm/jobs/nanotron_qwen25_instruction_sft_2026-06-24.slurm",
        "scripts/slurm/jobs/nanotron_qwen25_downstream_eval_2026-06-24.slurm",
        "scripts/slurm/jobs/nanotron_qwen25_multihop_eval_2026-07-13.slurm",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "scripts/nanotron/verify_qwen2_rope_config.py --checkpoint" in text
