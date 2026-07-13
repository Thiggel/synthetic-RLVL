from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "audit_nanotron_downstream_eval.py"
SPEC = importlib.util.spec_from_file_location("audit_nanotron_downstream_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_bundle(tmp_path: Path, *, mode: str = "direct", limit=None) -> Path:
    run_dir = tmp_path / mode
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    expects_chat = mode != "direct"
    command = ["python", "-m", "lm_eval", "run"]
    if expects_chat:
        command.append("--apply_chat_template")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    (run_dir / "command.json").write_text(
        json.dumps({"tasks": list(AUDIT.FINAL_TASKS), "cmd": command})
    )

    results = {}
    groups = {}
    leaf_tasks = []
    for task, metric in AUDIT.PRIMARY_METRICS.items():
        target = groups if task in {"bbh", "mmlu", "mmlu_pro"} else results
        target[task] = {metric: 0.5}
        if task not in {"bbh", "mmlu", "mmlu_pro", "mmlu_formal_logic"}:
            leaf_tasks.append(task)
    results["mmlu_formal_logic"] = {"acc,none": 0.5}
    leaf_tasks.extend(["bbh_test", "mmlu_formal_logic", "mmlu_pro_test"])
    n_samples = {task: {"original": 1, "effective": 1} for task in leaf_tasks}
    payload = {
        "results": results,
        "groups": groups,
        "n-samples": n_samples,
        "config": {"limit": limit},
        "chat_template": "template" if expects_chat else None,
    }
    (model_dir / "results_test.json").write_text(json.dumps(payload))

    prefix = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n" if expects_chat else "question"
    for task in leaf_tasks:
        row = {"doc_id": 0, "arguments": {"gen_args_0": {"arg_0": prefix}}}
        if task == "hendrycks_math500":
            row.update(
                {
                    "doc": {"answer": "4"},
                    "target": "4",
                    "filtered_resps": [" 4\nSolution: four."],
                    "filter": "none",
                    "exact_match": 0,
                }
            )
        (model_dir / f"samples_{task}_2026.jsonl").write_text(json.dumps(row) + "\n")
    return run_dir


def test_accepts_complete_direct_bundle(tmp_path: Path):
    run_dir = _write_bundle(tmp_path)
    report = AUDIT.audit_run(
        run_dir,
        mode="direct",
        expected_tasks=list(AUDIT.FINAL_TASKS),
    )
    assert report["accepted"]
    assert report["sample_file_count"] == 10
    assert report["math500_posthoc"]["accuracy"] == 1.0


def test_accepts_complete_instruction_bundle(tmp_path: Path):
    run_dir = _write_bundle(tmp_path, mode="instruction")
    report = AUDIT.audit_run(
        run_dir,
        mode="instruction",
        expected_tasks=list(AUDIT.FINAL_TASKS),
    )
    assert report["accepted"]
    assert report["chat_template_applied"]


def test_accepts_equivalent_task_order(tmp_path: Path):
    run_dir = _write_bundle(tmp_path)
    command_path = run_dir / "command.json"
    command = json.loads(command_path.read_text())
    command["tasks"] = list(reversed(command["tasks"]))
    command_path.write_text(json.dumps(command))
    report = AUDIT.audit_run(
        run_dir,
        mode="direct",
        expected_tasks=list(AUDIT.FINAL_TASKS),
    )
    assert report["accepted"]


def test_rejects_limited_production_bundle(tmp_path: Path):
    run_dir = _write_bundle(tmp_path, limit=1)
    report = AUDIT.audit_run(
        run_dir,
        mode="direct",
        expected_tasks=list(AUDIT.FINAL_TASKS),
    )
    assert not report["accepted"]
    assert any("limit" in error for error in report["errors"])


def test_rejects_missing_leaf_sample_file(tmp_path: Path):
    run_dir = _write_bundle(tmp_path)
    next(run_dir.rglob("samples_mmlu_formal_logic_*.jsonl")).unlink()
    report = AUDIT.audit_run(
        run_dir,
        mode="direct",
        expected_tasks=list(AUDIT.FINAL_TASKS),
    )
    assert not report["accepted"]
    assert any("mmlu_formal_logic" in error for error in report["errors"])


def test_rejects_unrendered_instruction_prompt(tmp_path: Path):
    run_dir = _write_bundle(tmp_path, mode="instruction")
    sample = next(run_dir.rglob("samples_gsm8k_*.jsonl"))
    sample.write_text(
        json.dumps({"doc_id": 0, "arguments": {"gen_args_0": {"arg_0": "question"}}})
        + "\n"
    )
    report = AUDIT.audit_run(
        run_dir,
        mode="instruction",
        expected_tasks=list(AUDIT.FINAL_TASKS),
    )
    assert not report["accepted"]
    assert any("retained prompt" in error for error in report["errors"])
