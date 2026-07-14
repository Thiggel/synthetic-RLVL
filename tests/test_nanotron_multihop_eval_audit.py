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
        tagged = task.endswith("_tagged")
        if tagged:
            prompt = (
                "<question>\nAnswer the question using the given passages. "
                "Put only the final answer in <answer>...</answer>.\n\n"
                "Passages:\nPassage 1:\nText.\n\nQuestion: Who?\n</question>\n"
            )
        else:
            prompt = (
                "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                "The following are given passages.\nPassage 1:\nText.\n\n"
                "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                "Question: Who?\nAnswer:"
            )
        rows = [
            {
                "doc_id": doc_id,
                "arguments": {
                    "gen_args_0": {
                        "arg_0": prompt,
                        "arg_1": {"max_gen_toks": 64 if tagged else 32},
                    }
                },
            }
            for doc_id in (0, 1)
        ]
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


def test_accepts_instruction_prompts_inside_qwen_chat_template(tmp_path: Path) -> None:
    task = "synthrlvl_longbench_hotpotqa_standard"
    command = {
        "tasks": [task],
        "cmd": [
            "python",
            "-m",
            "lm_eval",
            "--model_args",
            "max_model_len=32768",
            "--include_path",
            "lm_eval_tasks/synthrlvl_ood",
            "--apply_chat_template",
        ],
    }
    (tmp_path / "command.json").write_text(json.dumps(command), encoding="utf-8")
    payload = {
        "results": {task: {"qa_f1_score,none": 0.5}},
        "n-samples": {task: {"original": 1, "effective": 1}},
        "chat_template": "qwen",
        "config": {"limit": None},
    }
    (tmp_path / "results_2026-07-14.json").write_text(json.dumps(payload), encoding="utf-8")
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\nPassage 1:\nText.\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: Who?\nAnswer:"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    row = {
        "doc_id": 0,
        "arguments": {
            "gen_args_0": {
                "arg_0": prompt,
                "arg_1": {"max_gen_toks": 32},
            }
        },
    }
    (tmp_path / f"samples_{task}_2026-07-14.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    report = MODULE.audit(tmp_path, mode="instruction", expected_tasks=[task], require_full=True)

    assert report["accepted"]


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


def test_rejects_old_nested_tagged_prompt(tmp_path: Path) -> None:
    tasks = ["synthrlvl_longbench_hotpotqa_tagged"]
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
    payload = {
        "results": {
            tasks[0]: {
                "qa_f1_score,none": 0.0,
                "qa_exact_match,none": 0.0,
                "tag_found,none": 1.0,
                "extracted_nonempty,none": 1.0,
            }
        },
        "n-samples": {tasks[0]: {"original": 1, "effective": 1}},
        "chat_template": None,
        "config": {"limit": None},
    }
    (tmp_path / "results_2026-07-14.json").write_text(json.dumps(payload), encoding="utf-8")
    row = {
        "doc_id": 0,
        "arguments": {
            "gen_args_0": {
                "arg_0": (
                    "<question>\nPassages:\nAnswer the question based on the given passages. "
                    "The following are given passages.\nText.\nQuestion: Question: Who?\n</question>"
                ),
                "arg_1": {"max_gen_toks": 512},
            }
        },
    }
    (tmp_path / f"samples_{tasks[0]}_2026-07-14.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    report = MODULE.audit(tmp_path, mode="direct", expected_tasks=tasks, require_full=True)

    assert not report["accepted"]
    assert any("embedded stock wrapper" in error for error in report["errors"])
    assert any("max_gen_toks=512" in error for error in report["errors"])
