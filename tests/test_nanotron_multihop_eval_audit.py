from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "audit_nanotron_multihop_eval.py"
SPEC = importlib.util.spec_from_file_location("audit_nanotron_multihop_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_accepts_complete_direct_multihop_bundle(tmp_path: Path) -> None:
    tasks = list(MODULE.DEFAULT_TASKS)
    command = {
        "tasks": tasks,
        "cmd": [
            "python",
            "-m",
            "lm_eval",
            "--model_args",
            "max_model_len=32768",
            "--include_path",
            "lm_eval_tasks/synthrlvl_ood",
        ],
    }
    (tmp_path / "command.json").write_text(json.dumps(command), encoding="utf-8")

    results = {}
    n_samples = {}
    for task in tasks:
        task_results = {
            "qa_f1_score,none": 0.5,
        }
        if task.endswith("_tagged"):
            task_results.update(
                {
                    "qa_exact_match,none": 0.25,
                    "tag_found,none": 1.0,
                    "extracted_nonempty,none": 1.0,
                }
            )
        results[task] = task_results
        n_samples[task] = {"original": 2, "effective": 2}
        rows = [{"doc_id": 0}, {"doc_id": 1}]
        path = tmp_path / f"samples_{task}_2026-07-13.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    payload = {
        "results": results,
        "n-samples": n_samples,
        "chat_template": None,
        "config": {"limit": None},
    }
    (tmp_path / "results_2026-07-13.json").write_text(json.dumps(payload), encoding="utf-8")

    report = MODULE.audit(tmp_path, mode="direct", expected_tasks=tasks, require_full=True)
    assert report["accepted"]
    assert report["sample_row_count"] == 12


def test_rejects_truncating_model_window(tmp_path: Path) -> None:
    tasks = list(MODULE.DEFAULT_TASKS)
    command = {
        "tasks": tasks,
        "cmd": [
            "python",
            "-m",
            "lm_eval",
            "--model_args",
            "max_model_len=8192",
            "--include_path",
            "lm_eval_tasks/synthrlvl_ood",
        ],
    }
    (tmp_path / "command.json").write_text(json.dumps(command), encoding="utf-8")
    report = MODULE.audit(tmp_path, mode="direct", expected_tasks=tasks, require_full=True)
    assert not report["accepted"]
    assert any("max_model_len=[8192]" in error for error in report["errors"])
