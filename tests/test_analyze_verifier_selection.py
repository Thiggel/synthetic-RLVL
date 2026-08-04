import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "analyze_verifier_selection.py"
SPEC = importlib.util.spec_from_file_location("analyze_verifier_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
analyze_file = MODULE.analyze_file
summarize = MODULE.summarize


def _write_samples(path: Path, modality: str) -> None:
    rows = []
    validity_key = "citation_free_valid" if modality == "logic" else "nl_logic_citation_free_valid"
    fraction_key = (
        "citation_free_line_valid_fraction"
        if modality == "logic"
        else "nl_logic_line_valid_fraction"
    )
    candidates = [
        (False, False, 0.2),
        (True, True, 1.0),
        (False, True, 1.0),
    ]
    for prompt, step in (("id", 25), ("ood", 30)):
        for correct, valid, fraction in candidates:
            rows.append(
                {
                    "source": "synthetic_sampled",
                    "step": step,
                    "prompt": prompt,
                    "correct": float(correct),
                    validity_key: float(valid),
                    fraction_key: fraction,
                }
            )
    rows.append({"source": "synthetic", "step": 30, "prompt": "ignored"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_verifier_selection_uses_validity_without_gold(tmp_path: Path) -> None:
    path = tmp_path / "run_logic_train1to25_seed3407_samples.jsonl"
    _write_samples(path, "logic")

    result = analyze_file(path, expected_k=3, train_max=25)

    assert result["prompt_groups"] == 2
    assert result["metrics_by_band"]["ood"]["random_correct"] == 0.0
    assert result["metrics_by_band"]["ood"]["first_valid_correct"] == 1.0
    assert result["metrics_by_band"]["ood"]["oracle_joint"] == 1.0


def test_summary_reports_population_seed_statistics(tmp_path: Path) -> None:
    logic = tmp_path / "run_logic_train1to25_seed3407_samples.jsonl"
    natural = tmp_path / "run_nl_exact_train1to25_seed3408_samples.jsonl"
    _write_samples(logic, "logic")
    _write_samples(natural, "nl_exact")

    summary = summarize(
        [
            analyze_file(logic, expected_k=3, train_max=25),
            analyze_file(natural, expected_k=3, train_max=25),
        ]
    )

    assert summary["logic"]["ood"]["n_seeds"] == 1
    assert summary["logic"]["ood"]["first_valid_correct"]["mean"] == 1.0
    assert summary["nl_exact"]["ood"]["first_valid_correct"]["std"] == 0.0


def test_rejects_incomplete_prompt_groups(tmp_path: Path) -> None:
    path = tmp_path / "run_logic_train1to25_seed3407_samples.jsonl"
    _write_samples(path, "logic")

    try:
        analyze_file(path, expected_k=16, train_max=25)
    except ValueError as exc:
        assert "expected 16" in str(exc)
    else:
        raise AssertionError("incomplete prompt group was accepted")
