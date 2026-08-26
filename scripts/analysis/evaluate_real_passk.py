#!/usr/bin/env python3
"""Pass@k / majority-vote evaluation on real benchmarks for Dolmino post-SFT checkpoints.

Replays the exact prompts recorded in the accepted greedy lm-eval sample files
(`lm_eval_results/qwen25_dolmino_post_sft_20260804`) through vLLM with
n=16 sampled generations (T=0.8, top_p=0.95) plus one greedy generation, and
scores with the same extractors/scorers the accepted evals used:

- gsm8k: lm-eval strict-match ("#### N") and flexible-extract regex filters,
  exact_match with the stock regexes_to_ignore.
- hendrycks_math500: answer-prefix extraction + math_verify equivalence,
  imported from scripts/analysis/rescore_math500.py.
- synthrlvl_longbench_{hotpotqa,2wikimqa,musique}_tagged: QA-F1/EM against all
  gold answers with strict <answer>-tag extraction (imported from
  lm_eval_tasks/synthrlvl_ood/utils.py) plus a fallback extractor (first
  non-empty response line when no tag).

Metrics per task: greedy accuracy, unbiased pass@k (k in 1,2,4,8,16), maj@k
(k in 1,4,8,16; ties broken by first occurrence), extraction-failure rate,
mean response length. Outputs (per condition, a handful of files only):
samples_<task>.jsonl, metrics_<task>.json, audit_<group>.json.

Subcommands:
  run        one condition x task-group on one GPU
  summarize  aggregate metrics JSONs across conditions into JSON + markdown
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

CONDITIONS = ("control", "logic", "nl_exact")
RUN_NAME_TEMPLATE = "qwen25_7b_dolmino_{condition}_5b_dolci_100k_lr5em6"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/home/atuin/c107fa/c107fa12/synthetic-RLVL/post_sft_dolci_20260804"
)
DEFAULT_ACCEPTED_ROOT = Path(
    "/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_20260804"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806"
)

N_SAMPLES = 16
PASS_KS = (1, 2, 4, 8, 16)
MAJ_KS = (1, 4, 8, 16)
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.95
DEFAULT_SEED = 20260806
SCHEMA_VERSION = 1

TASK_GROUPS = {
    "standard": {
        "suite_suffix": "standard",
        "max_model_len": 8192,
        "tasks": {
            "gsm8k": {
                "expected": 1319,
                "stop": ["Question:", "</s>", "<|im_end|>"],
                "greedy_max_tokens": 256,  # lm-eval default max_gen_toks (accepted run)
                "sampled_max_tokens": 512,  # headroom so T=0.8 samples are not truncation-biased
            },
            "hendrycks_math500": {
                "expected": 500,
                "stop": ["Problem:", "</s>", "<|im_end|>"],
                "greedy_max_tokens": 256,
                "sampled_max_tokens": 512,
            },
        },
    },
    "multihop": {
        "suite_suffix": "multihop",
        "max_model_len": 32768,
        "tasks": {
            "synthrlvl_longbench_hotpotqa_tagged": {
                "expected": 200,
                "stop": ["</answer>", "<|im_end|>", "</s>"],
                "greedy_max_tokens": 4096,  # matches raised task yaml cap (2026-08-25 capfix)
                "sampled_max_tokens": 4096,
            },
            "synthrlvl_longbench_2wikimqa_tagged": {
                "expected": 200,
                "stop": ["</answer>", "<|im_end|>", "</s>"],
                "greedy_max_tokens": 4096,  # matches raised task yaml cap (2026-08-25 capfix)
                "sampled_max_tokens": 4096,
            },
            "synthrlvl_longbench_musique_tagged": {
                "expected": 200,
                "stop": ["</answer>", "<|im_end|>", "</s>"],
                "greedy_max_tokens": 4096,  # matches raised task yaml cap (2026-08-25 capfix)
                "sampled_max_tokens": 4096,
            },
        },
    },
}

MAX_EMPTY_GREEDY_RATE = 0.02
MAX_EMPTY_SAMPLED_RATE = 0.05


# ---------------------------------------------------------------------------
# Imported scorers (exactly the accepted implementations)
# ---------------------------------------------------------------------------


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_math500 = _load_module("rescore_math500_imported", SCRIPT_DIR / "rescore_math500.py")
_mh = _load_module(
    "synthrlvl_ood_utils_imported", REPO_ROOT / "lm_eval_tasks" / "synthrlvl_ood" / "utils.py"
)


# ---------------------------------------------------------------------------
# GSM8K extraction/scoring (replicates lm-eval gsm8k yaml filters + exact_match)
# ---------------------------------------------------------------------------

_GSM8K_STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_GSM8K_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_GSM8K_IGNORE_RES = [re.compile(p, re.DOTALL) for p in (",", r"\$", r"(?s).*#### ", r"\.$")]
_GSM8K_INVALID = "[invalid]"


def gsm8k_extract_strict(response: str) -> str:
    matches = _GSM8K_STRICT_RE.findall(response)
    return matches[0] if matches else _GSM8K_INVALID


def gsm8k_extract_flexible(response: str) -> str:
    matches = _GSM8K_FLEX_RE.findall(response)
    if not matches:
        return _GSM8K_INVALID
    last = matches[-1]
    if isinstance(last, tuple):
        nonempty = [part for part in last if part]
        return nonempty[0] if nonempty else _GSM8K_INVALID
    return last or _GSM8K_INVALID


def gsm8k_normalize(text: str) -> str:
    for pattern in _GSM8K_IGNORE_RES:
        text = pattern.sub("", text)
    return text.strip().lower()


# ---------------------------------------------------------------------------
# Generic metric helpers
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if not 0 <= c <= n or k > n:
        raise ValueError(f"invalid pass@k args n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def expected_max_at_k(values: list[float], k: int) -> float:
    """Exact expectation of max(value) over a uniformly random size-k subset."""
    n = len(values)
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    ordered = sorted(values)
    total = math.comb(n, k)
    acc = 0.0
    for i in range(k, n + 1):  # i = 1-based rank of the max within sorted order
        weight = math.comb(i - 1, k - 1) / total
        acc += ordered[i - 1] * weight
    return acc


def majority_key(keys: list[str | None], k: int) -> int | None:
    """Index (into keys) of the majority answer among the first k entries.

    Failed extractions (None) do not vote. Ties are broken by the first
    occurrence of the tied answer. Returns None when nothing voted.
    """
    votes: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for index, key in enumerate(keys[:k]):
        if key is None:
            continue
        votes[key] = votes.get(key, 0) + 1
        first_index.setdefault(key, index)
    if not votes:
        return None
    winner = min(votes, key=lambda key: (-votes[key], first_index[key]))
    return first_index[winner]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def assert_finite_metrics(obj: Any, path: str = "metrics") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_finite_metrics(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            assert_finite_metrics(value, f"{path}[{index}]")
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    elif not _finite(obj):
        raise AssertionError(f"non-finite metric at {path}: {obj!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Task scorers: response -> record
# ---------------------------------------------------------------------------


class Gsm8kScorer:
    name = "gsm8k"

    def score(self, doc: dict, target: str, response: str) -> dict[str, Any]:
        strict = gsm8k_extract_strict(response)
        flexible = gsm8k_extract_flexible(response)
        gold = gsm8k_normalize(target)
        strict_ok = gsm8k_normalize(strict) == gold if strict != _GSM8K_INVALID else False
        flexible_ok = gsm8k_normalize(flexible) == gold if flexible != _GSM8K_INVALID else False
        return {
            "extracted": None if strict == _GSM8K_INVALID else strict,
            "extracted_flexible": None if flexible == _GSM8K_INVALID else flexible,
            "correct": bool(strict_ok),
            "correct_flexible": bool(flexible_ok),
            "maj_key": gsm8k_normalize(strict) if strict != _GSM8K_INVALID else None,
            "extraction_failed": strict == _GSM8K_INVALID,
        }



class Math500Scorer:
    name = "hendrycks_math500"

    def score(self, doc: dict, target: str, response: str) -> dict[str, Any]:
        candidate, extraction = _math500.extract_answer_prefix(target, response)
        correct, score_error = _math500.equivalent(target, candidate)
        return {
            "extracted": candidate,
            "extraction": extraction,
            "correct": bool(correct),
            "score_error": score_error,
            "maj_key": " ".join(candidate.split()) if candidate is not None else None,
            "extraction_failed": candidate is None,
        }



class MultihopScorer:
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def _fallback_extract(response: str) -> str:
        strict = _mh.extract_answer(response, allow_raw_fallback=False)
        if strict.strip():
            return strict
        first_line = next(
            (line.strip() for line in str(response).splitlines() if line.strip()), ""
        )
        return _mh._clean_extracted(first_line)

    @staticmethod
    def _best_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
        best_f1 = 0.0
        best_em = 0.0
        for answer in answers:
            best_f1 = max(best_f1, _mh.qa_f1_score(prediction, str(answer)))
            best_em = max(best_em, _mh.qa_exact_match(prediction, str(answer)))
        return best_f1, best_em

    def score(self, doc: dict, target: str, response: str) -> dict[str, Any]:
        answers = [str(a) for a in doc["answers"]]
        strict = _mh.extract_answer(response, allow_raw_fallback=False)
        fallback = self._fallback_extract(response)
        strict_f1, strict_em = self._best_scores(strict, answers)
        fb_f1, fb_em = self._best_scores(fallback, answers)
        tag_found = bool(_mh._ANSWER_RE.search(str(response)))
        return {
            "extracted": strict if strict.strip() else None,
            "extracted_fallback": fallback if fallback.strip() else None,
            "tag_found": tag_found,
            "f1": float(strict_f1),
            "em": float(strict_em),
            "f1_fallback": float(fb_f1),
            "em_fallback": float(fb_em),
            "correct": bool(strict_em),
            "maj_key": _mh.normalize_answer(strict) if strict.strip() else None,
            "maj_key_fallback": _mh.normalize_answer(fallback) if fallback.strip() else None,
            "extraction_failed": not strict.strip(),
        }



def make_scorer(task: str):
    if task == "gsm8k":
        return Gsm8kScorer()
    if task == "hendrycks_math500":
        return Math500Scorer()
    if task.startswith("synthrlvl_longbench_") and task.endswith("_tagged"):
        return MultihopScorer(task)
    raise ValueError(f"no scorer for task {task}")


# ---------------------------------------------------------------------------
# Prompt loading from accepted lm-eval sample files
# ---------------------------------------------------------------------------


def find_samples_file(accepted_run_dir: Path, task: str) -> Path:
    pattern = str(accepted_run_dir / "__home__*" / f"samples_{task}_*.jsonl")
    paths = sorted(glob.glob(pattern))
    if len(paths) != 1:
        raise AssertionError(f"expected exactly one samples file for {pattern}, found {len(paths)}")
    return Path(paths[0])


def load_docs(accepted_run_dir: Path, task: str, expected: int, limit: int | None) -> list[dict]:
    path = find_samples_file(accepted_run_dir, task)
    docs: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row["doc_id"]
            prompt = row["arguments"]["gen_args_0"]["arg_0"]
            assert isinstance(prompt, str) and prompt, f"{task} doc {doc_id}: empty prompt"
            entry = {
                "doc_id": doc_id,
                "prompt": prompt,
                "target": row.get("target", ""),
                "doc": row.get("doc", {}),
            }
            if doc_id in docs:
                # gsm8k accepted files carry one row per filter; prompts must agree.
                assert docs[doc_id]["prompt"] == prompt, f"{task} doc {doc_id}: prompt mismatch"
            else:
                docs[doc_id] = entry
    ordered = [docs[doc_id] for doc_id in sorted(docs)]
    assert len(ordered) == expected, (
        f"{task}: expected {expected} unique docs in {path}, found {len(ordered)}"
    )
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def prepare_token_prompts(tokenizer, docs, max_model_len: int, max_new_tokens: int):
    """Tokenize recorded prompt strings; left-truncate like lm-eval when too long."""
    budget = max_model_len - max_new_tokens - 1
    token_prompts = []
    truncated = 0
    for doc in docs:
        ids = tokenizer(doc["prompt"], add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            ids = ids[-budget:]
            truncated += 1
        token_prompts.append(ids)
    return token_prompts, truncated


def generate_for_task(llm, sampling_params_cls, token_prompts, task_cfg, seed: int):
    greedy_params = sampling_params_cls(
        n=1,
        temperature=0.0,
        max_tokens=task_cfg["greedy_max_tokens"],
        stop=list(task_cfg["stop"]),
        seed=seed,
    )
    sampled_params = sampling_params_cls(
        n=N_SAMPLES,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        max_tokens=task_cfg["sampled_max_tokens"],
        stop=list(task_cfg["stop"]),
        seed=seed,
    )
    prompts = [{"prompt_token_ids": ids} for ids in token_prompts]
    greedy_out = llm.generate(prompts, greedy_params)
    sampled_out = llm.generate(prompts, sampled_params)
    assert len(greedy_out) == len(prompts) and len(sampled_out) == len(prompts)
    return greedy_out, sampled_out


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


def evaluate_task(task, task_cfg, docs, greedy_out, sampled_out, out_dir: Path, meta: dict):
    scorer = make_scorer(task)
    rows = []
    for doc, greedy_req, sampled_req in zip(docs, greedy_out, sampled_out):
        assert len(greedy_req.outputs) == 1, f"{task} doc {doc['doc_id']}: greedy n != 1"
        assert len(sampled_req.outputs) == N_SAMPLES, (
            f"{task} doc {doc['doc_id']}: expected {N_SAMPLES} samples, "
            f"got {len(sampled_req.outputs)}"
        )
        greedy_text = greedy_req.outputs[0].text
        greedy_rec = scorer.score(doc["doc"], doc["target"], greedy_text)
        greedy_rec["text"] = greedy_text
        greedy_rec["n_tokens"] = len(greedy_req.outputs[0].token_ids)

        sample_recs = []
        for output in sorted(sampled_req.outputs, key=lambda o: o.index):
            rec = scorer.score(doc["doc"], doc["target"], output.text)
            rec["text"] = output.text
            rec["n_tokens"] = len(output.token_ids)
            sample_recs.append(rec)

        rows.append(
            {
                "task": task,
                "doc_id": doc["doc_id"],
                "prompt_hash": sha256_text(doc["prompt"]),
                "target": doc["target"],
                "gold_answers": [str(a) for a in doc["doc"].get("answers", [])] or None,
                "greedy": greedy_rec,
                "samples": sample_recs,
            }
        )

    # ---- aggregate metrics -------------------------------------------------
    n_docs = len(rows)
    assert n_docs > 0, f"{task}: no rows"

    def rate(flags):
        return sum(bool(f) for f in flags) / len(flags)

    greedy_recs = [row["greedy"] for row in rows]
    all_sample_recs = [rec for row in rows for rec in row["samples"]]

    empty_greedy = sum(not rec["text"].strip() for rec in greedy_recs)
    empty_sampled = sum(not rec["text"].strip() for rec in all_sample_recs)
    assert empty_greedy / n_docs <= MAX_EMPTY_GREEDY_RATE, (
        f"{task}: {empty_greedy}/{n_docs} empty greedy generations"
    )
    assert empty_sampled / len(all_sample_recs) <= MAX_EMPTY_SAMPLED_RATE, (
        f"{task}: {empty_sampled}/{len(all_sample_recs)} empty sampled generations"
    )

    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "n_docs": n_docs,
        "n_samples_per_doc": N_SAMPLES,
        **meta,
        "greedy": {
            "accuracy": rate([r["correct"] for r in greedy_recs]),
            "extraction_failure_rate": rate([r["extraction_failed"] for r in greedy_recs]),
            "mean_response_tokens": statistics.fmean(r["n_tokens"] for r in greedy_recs),
            "mean_response_chars": statistics.fmean(len(r["text"]) for r in greedy_recs),
        },
        "sampled": {
            "mean_accuracy": rate([r["correct"] for r in all_sample_recs]),
            "extraction_failure_rate": rate([r["extraction_failed"] for r in all_sample_recs]),
            "mean_response_tokens": statistics.fmean(r["n_tokens"] for r in all_sample_recs),
            "mean_response_chars": statistics.fmean(len(r["text"]) for r in all_sample_recs),
            "empty_rate": empty_sampled / len(all_sample_recs),
        },
    }
    if task == "gsm8k":
        metrics["greedy"]["accuracy_flexible"] = rate([r["correct_flexible"] for r in greedy_recs])
        metrics["sampled"]["mean_accuracy_flexible"] = rate(
            [r["correct_flexible"] for r in all_sample_recs]
        )
    if isinstance(scorer, MultihopScorer):
        metrics["greedy"]["qa_f1"] = statistics.fmean(r["f1"] for r in greedy_recs)
        metrics["greedy"]["qa_f1_fallback"] = statistics.fmean(
            r["f1_fallback"] for r in greedy_recs
        )
        metrics["greedy"]["accuracy_fallback"] = rate([r["em_fallback"] for r in greedy_recs])
        metrics["greedy"]["tag_rate"] = rate([r["tag_found"] for r in greedy_recs])
        metrics["sampled"]["qa_f1"] = statistics.fmean(r["f1"] for r in all_sample_recs)
        metrics["sampled"]["qa_f1_fallback"] = statistics.fmean(
            r["f1_fallback"] for r in all_sample_recs
        )
        metrics["sampled"]["tag_rate"] = rate([r["tag_found"] for r in all_sample_recs])

    # pass@k over binary per-sample success
    def pass_curve(flag_key: str) -> dict[str, float]:
        curve = {}
        for k in PASS_KS:
            values = []
            for row in rows:
                c = sum(bool(rec[flag_key]) for rec in row["samples"])
                values.append(pass_at_k(N_SAMPLES, c, k))
            curve[str(k)] = statistics.fmean(values)
        return curve

    metrics["pass_at_k"] = pass_curve("correct")
    if task == "gsm8k":
        metrics["pass_at_k_flexible"] = pass_curve("correct_flexible")
    if isinstance(scorer, MultihopScorer):
        metrics["pass_at_k_fallback"] = pass_curve("em_fallback")
        metrics["best_f1_at_k"] = {
            str(k): statistics.fmean(
                expected_max_at_k([rec["f1"] for rec in row["samples"]], k) for row in rows
            )
            for k in PASS_KS
        }
        metrics["best_f1_at_k_fallback"] = {
            str(k): statistics.fmean(
                expected_max_at_k([rec["f1_fallback"] for rec in row["samples"]], k)
                for row in rows
            )
            for k in PASS_KS
        }

    # maj@k (first-k prefix vote; ties broken by first occurrence)
    def maj_curve(key_field: str, correct_of) -> dict[str, float]:
        curve = {}
        for k in MAJ_KS:
            hits = 0.0
            for row in rows:
                keys = [rec.get(key_field) for rec in row["samples"]]
                winner = majority_key(keys, k)
                if winner is not None:
                    hits += correct_of(row, winner)
            curve[str(k)] = hits / n_docs
        return curve

    metrics["maj_at_k"] = maj_curve(
        "maj_key", lambda row, i: float(row["samples"][i]["correct"])
    )
    if isinstance(scorer, MultihopScorer):
        metrics["maj_at_k_f1"] = maj_curve(
            "maj_key", lambda row, i: float(row["samples"][i]["f1"])
        )
        metrics["maj_at_k_fallback"] = maj_curve(
            "maj_key_fallback", lambda row, i: float(row["samples"][i]["em_fallback"])
        )
        metrics["maj_at_k_f1_fallback"] = maj_curve(
            "maj_key_fallback", lambda row, i: float(row["samples"][i]["f1_fallback"])
        )

    assert_finite_metrics(
        {k: v for k, v in metrics.items() if k not in ("task",)}, f"metrics[{task}]"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"samples_{task}.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path = out_dir / f"metrics_{task}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    audit_entry = {
        "n_docs": n_docs,
        "expected": task_cfg["expected"],
        "empty_greedy": empty_greedy,
        "empty_sampled": empty_sampled,
        "greedy_accuracy": metrics["greedy"]["accuracy"],
        "pass_at_16": metrics["pass_at_k"]["16"],
        "samples_file": str(samples_path),
        "samples_sha256": _math500._sha256(samples_path),
        "metrics_file": str(metrics_path),
    }
    return metrics, audit_entry


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


def cmd_run(args) -> None:
    condition = args.condition
    group = args.task_group
    group_cfg = TASK_GROUPS[group]
    run_name = RUN_NAME_TEMPLATE.format(condition=condition)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        DEFAULT_CHECKPOINT_ROOT / run_name / "final"
    )
    accepted_run_dir = Path(args.accepted_run_dir) if args.accepted_run_dir else (
        DEFAULT_ACCEPTED_ROOT / f"{run_name}_{group_cfg['suite_suffix']}"
    )
    out_dir = Path(args.output_root) / condition

    config = json.loads((checkpoint / "config.json").read_text())
    rope_theta = config.get("rope_theta")
    assert rope_theta == 1000000 or rope_theta == 1000000.0, (
        f"{checkpoint}: rope_theta={rope_theta!r}, expected 1000000"
    )

    tasks = {
        task: load_docs(accepted_run_dir, task, cfg["expected"], args.limit)
        for task, cfg in group_cfg["tasks"].items()
    }

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import vllm

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    llm = LLM(
        model=str(checkpoint),
        dtype="bfloat16",
        max_model_len=group_cfg["max_model_len"],
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        enable_prefix_caching=True,
    )

    meta = {
        "condition": condition,
        "task_group": group,
        "checkpoint": str(checkpoint),
        "accepted_run_dir": str(accepted_run_dir),
        "rope_theta": rope_theta,
        "max_model_len": group_cfg["max_model_len"],
        "sampling": {
            "n": N_SAMPLES,
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "seed": args.seed,
        },
        "limit": args.limit,
        "vllm_version": vllm.__version__,
    }

    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        **meta,
        "tasks": {},
    }
    for task, docs in tasks.items():
        task_cfg = group_cfg["tasks"][task]
        max_new = max(task_cfg["greedy_max_tokens"], task_cfg["sampled_max_tokens"])
        token_prompts, truncated = prepare_token_prompts(
            tokenizer, docs, group_cfg["max_model_len"], max_new
        )
        greedy_out, sampled_out = generate_for_task(
            llm, SamplingParams, token_prompts, task_cfg, args.seed
        )
        metrics, audit_entry = evaluate_task(
            task, task_cfg, docs, greedy_out, sampled_out, out_dir, meta
        )
        audit_entry["prompt_truncated"] = truncated
        audit["tasks"][task] = audit_entry
        print(f"[{condition}/{task}] greedy={metrics['greedy']['accuracy']:.4f} "
              f"pass@16={metrics['pass_at_k']['16']:.4f} maj@16={metrics['maj_at_k']['16']:.4f}")

    audit["accepted"] = bool(
        args.limit is None
        and all(
            entry["n_docs"] == entry["expected"] for entry in audit["tasks"].values()
        )
    )
    audit_path = out_dir / f"audit_{group}.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"wrote audit {audit_path} accepted={audit['accepted']}")


# ---------------------------------------------------------------------------
# summarize subcommand
# ---------------------------------------------------------------------------


def cmd_summarize(args) -> None:
    output_root = Path(args.output_root)
    conditions = tuple(args.conditions.split(",")) if args.conditions else CONDITIONS
    all_metrics: dict[str, dict[str, dict]] = {}
    for condition in conditions:
        for path in sorted((output_root / condition).glob("metrics_*.json")):
            metrics = json.loads(path.read_text())
            all_metrics.setdefault(metrics["task"], {})[condition] = metrics

    if not all_metrics:
        raise SystemExit(f"no metrics found under {output_root}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "tasks": all_metrics,
    }
    (output_root / "summary_passk.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = ["# Real-benchmark pass@k / maj@k summary", ""]
    for task, by_condition in sorted(all_metrics.items()):
        lines.append(f"## {task}")
        lines.append("")
        header = (
            "| condition | greedy | " + " | ".join(f"pass@{k}" for k in PASS_KS)
            + " | " + " | ".join(f"maj@{k}" for k in MAJ_KS)
            + " | extr-fail (sampled) |"
        )
        lines.append(header)
        lines.append("|" + "---|" * (2 + len(PASS_KS) + len(MAJ_KS) + 1))
        for condition in conditions:
            metrics = by_condition.get(condition)
            if metrics is None:
                lines.append(f"| {condition} | (missing) |" + " |" * (len(PASS_KS) + len(MAJ_KS) + 1))
                continue
            cells = [f"{metrics['greedy']['accuracy']:.4f}"]
            cells += [f"{metrics['pass_at_k'][str(k)]:.4f}" for k in PASS_KS]
            cells += [f"{metrics['maj_at_k'][str(k)]:.4f}" for k in MAJ_KS]
            cells += [f"{metrics['sampled']['extraction_failure_rate']:.4f}"]
            lines.append(f"| {condition} | " + " | ".join(cells) + " |")
        lines.append("")
        if any("qa_f1" in by_condition[c]["sampled"] for c in by_condition):
            lines.append(
                "| condition | greedy F1 | bestF1@16 | majF1@16 | greedy F1 (fallback) | "
                "bestF1@16 (fallback) | majF1@16 (fallback) | tag rate (sampled) |"
            )
            lines.append("|" + "---|" * 8)
            for condition in conditions:
                metrics = by_condition.get(condition)
                if metrics is None:
                    continue
                lines.append(
                    f"| {condition} | {metrics['greedy']['qa_f1']:.4f} | "
                    f"{metrics['best_f1_at_k']['16']:.4f} | {metrics['maj_at_k_f1']['16']:.4f} | "
                    f"{metrics['greedy']['qa_f1_fallback']:.4f} | "
                    f"{metrics['best_f1_at_k_fallback']['16']:.4f} | "
                    f"{metrics['maj_at_k_f1_fallback']['16']:.4f} | "
                    f"{metrics['sampled']['tag_rate']:.4f} |"
                )
            lines.append("")
    (output_root / "summary_passk.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {output_root / 'summary_passk.json'} and summary_passk.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one condition x task-group")
    run.add_argument("--condition", required=True)
    run.add_argument("--task-group", required=True, choices=sorted(TASK_GROUPS))
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--accepted-run-dir", default=None)
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.set_defaults(func=cmd_run)

    summarize = sub.add_parser("summarize", help="aggregate metrics across conditions")
    summarize.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    summarize.add_argument(
        "--conditions",
        default=None,
        help="comma-separated condition labels (default: the dolmino three-way)",
    )
    summarize.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
