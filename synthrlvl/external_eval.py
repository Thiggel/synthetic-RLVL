from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from logic_engine import LogicEngine

from .metrics import extract_tag


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    loader: Callable[[int], list[tuple[str, str]]]


def _proofwriter_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("tasksource/proofwriter", split=f"validation[:{limit}]")
    rows = []
    for row in ds:
        theory = str(row.get("theory", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        prompt = f"Question: Theory:\n{theory}\n\nQuery:\n{question}\n\nAnswer with <answer>...</answer>.\nAnswer:"
        rows.append((prompt, answer))
    return rows


def _folio_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("tasksource/folio", split=f"validation[:{limit}]")
    rows = []
    for row in ds:
        premises = str(row.get("premises", "")).strip()
        hyp = str(row.get("conclusion", "")).strip()
        label = str(row.get("label", "")).strip()
        prompt = (
            "Question: Premises:\n"
            f"{premises}\n\nHypothesis:\n{hyp}\n\n"
            "Answer exactly True/False/Uncertain using <answer>...</answer>.\nAnswer:"
        )
        rows.append((prompt, label))
    return rows


def _prontoqa_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("renma/ProntoQA", split=f"validation[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        rows.append((f"Question: {q}\nAnswer with <answer>...</answer>.\nAnswer:", a))
    return rows


def _proverqa_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("TIGER-Lab/ProverQA", split=f"test[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        rows.append((f"Question: {q}\nAnswer with <answer>...</answer>.\nAnswer:", a))
    return rows


def _logiqa2_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("lucasmccabe/logiqa", "logiqa2", split=f"test[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        choices = row.get("options", [])
        gold = str(row.get("answer", "")).strip()
        choice_text = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
        prompt = f"Question: {q}\n{choice_text}\nRespond with <answer>A/B/C/D</answer>.\nAnswer:"
        rows.append((prompt, gold))
    return rows


def _bbh_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("lukaemon/bbh", "boolean_expressions", split=f"test[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("input", "")).strip()
        a = str(row.get("target", "")).strip()
        rows.append((f"Question: {q}\nAnswer with <answer>...</answer>.\nAnswer:", a))
    return rows


def _mmlu_philosophy_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("cais/mmlu", "philosophy", split=f"test[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        choices = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
        ans = str(row.get("answer", "")).strip()
        choice_text = "\n".join(f"{k}. {v}" for k, v in zip(["A", "B", "C", "D"], choices))
        rows.append((f"Question: {q}\n{choice_text}\nRespond with <answer>A/B/C/D</answer>.\nAnswer:", ans))
    return rows


def _mmlu_logic_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("cais/mmlu", "formal_logic", split=f"test[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        choices = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
        ans = str(row.get("answer", "")).strip()
        choice_text = "\n".join(f"{k}. {v}" for k, v in zip(["A", "B", "C", "D"], choices))
        rows.append((f"Question: {q}\n{choice_text}\nRespond with <answer>A/B/C/D</answer>.\nAnswer:", ans))
    return rows


def _strategyqa_loader(limit: int) -> list[tuple[str, str]]:
    ds = load_dataset("tasksource/strategyqa", split=f"validation[:{limit}]")
    rows = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        ans = str(row.get("answer", "")).strip()
        rows.append((f"Question: {q}\nAnswer yes or no in <answer>...</answer>.\nAnswer:", ans))
    return rows


SPECS = {
    "proofwriter": BenchmarkSpec("proofwriter", _proofwriter_loader),
    "folio": BenchmarkSpec("folio", _folio_loader),
    "prontoqa": BenchmarkSpec("prontoqa", _prontoqa_loader),
    "proverqa": BenchmarkSpec("proverqa", _proverqa_loader),
    "logiqa2": BenchmarkSpec("logiqa2", _logiqa2_loader),
    "bbh": BenchmarkSpec("bbh", _bbh_loader),
    "mmlu_philosophy": BenchmarkSpec("mmlu_philosophy", _mmlu_philosophy_loader),
    "mmlu_formal_logic": BenchmarkSpec("mmlu_formal_logic", _mmlu_logic_loader),
    "strategyqa": BenchmarkSpec("strategyqa", _strategyqa_loader),
}


def _batched(items: Sequence[tuple[str, str]], batch_size: int) -> list[Sequence[tuple[str, str]]]:
    bs = max(1, int(batch_size))
    return [items[i : i + bs] for i in range(0, len(items), bs)]


def _score_prediction_text(text: str, gold: str, engine: LogicEngine) -> tuple[float, float, float]:
    pred = extract_tag(text, "answer")
    if not pred:
        stripped = text.strip()
        pred = stripped.splitlines()[0].strip() if stripped else ""
    correct = float(pred.strip().lower() == str(gold).strip().lower())

    logic = extract_tag(text, "logic")
    if not logic:
        return float(bool(pred)), correct, 0.0
    premises = extract_tag(logic, "premises")
    proof = extract_tag(logic, "proof")
    conclusion = extract_tag(logic, "conclusion")
    if not (premises and proof and conclusion):
        return float(bool(pred)), correct, 0.0
    try:
        valid = float(engine.validate_proof(premises=premises, conclusion=conclusion, proof=proof))
    except Exception:
        valid = 0.0
    return float(bool(pred)), correct, valid


def evaluate_external_benchmarks_with_generate_fn(
    *,
    names: list[str],
    limit_per_benchmark: int,
    max_new_tokens: int,
    generate_texts: Callable[[Sequence[str], int], list[str]],
    batch_size: int = 1,
    collect_samples: int = 0,
) -> tuple[Dict[str, float], list[dict]]:
    out: Dict[str, float] = {}
    samples: list[dict] = []
    engine = LogicEngine()

    for name in names:
        spec = SPECS.get(name)
        if spec is None:
            out[f"external/{name}/acc"] = float("nan")
            continue
        try:
            rows = spec.loader(limit_per_benchmark)
        except Exception:
            out[f"external/{name}/acc"] = float("nan")
            continue

        hits = 0.0
        valid_hits = 0.0
        n = 0
        for chunk in _batched(rows, batch_size=batch_size):
            prompts = [p for p, _ in chunk]
            golds = [g for _, g in chunk]
            texts = generate_texts(prompts, max_new_tokens)
            if len(texts) != len(chunk):
                raise RuntimeError(f"external eval generate_texts returned {len(texts)} outputs for {len(chunk)} prompts")
            for prompt, gold, text in zip(prompts, golds, texts, strict=True):
                format_ok, correct, valid = _score_prediction_text(text, str(gold), engine)
                hits += correct
                valid_hits += valid
                if len(samples) < collect_samples:
                    samples.append(
                        {
                            "source": f"external:{name}",
                            "step": -1,
                            "prompt": prompt,
                            "generation": text,
                            "gold_answer": str(gold),
                            "format_ok": format_ok,
                            "correct": correct,
                            "valid": valid,
                        }
                    )
                n += 1

        out[f"external/{name}/acc"] = hits / max(1, n)
        out[f"external/{name}/valid"] = valid_hits / max(1, n)
    return out, samples


@torch.no_grad()
def evaluate_external_benchmarks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    names: list[str],
    limit_per_benchmark: int,
    max_new_tokens: int,
    device: torch.device,
    batch_size: int = 1,
    collect_samples: int = 0,
) -> tuple[Dict[str, float], list[dict]]:
    model.eval()

    def _hf_generate_texts(prompts: Sequence[str], max_tokens: int) -> list[str]:
        def _decode_batch(tok_batch, out_batch, expected: int) -> list[str]:
            if "attention_mask" in tok_batch:
                prompt_lens = tok_batch["attention_mask"].sum(dim=1).tolist()
            else:
                prompt_lens = [int(tok_batch["input_ids"].shape[1])] * expected
            return [
                tokenizer.decode(out_batch[i][int(prompt_lens[i]) :], skip_special_tokens=True)
                for i in range(expected)
            ]

        try:
            toks = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
            out_ids = model.generate(
                **toks,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            return _decode_batch(toks, out_ids, len(prompts))
        except TypeError:
            # Compatibility fallback for minimal tokenizers used in tests.
            outs: list[str] = []
            for prompt in prompts:
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                out_ids = model.generate(
                    **toks,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                outs.extend(_decode_batch(toks, out_ids, 1))
            return outs

    out, samples = evaluate_external_benchmarks_with_generate_fn(
        names=names,
        limit_per_benchmark=limit_per_benchmark,
        max_new_tokens=max_new_tokens,
        generate_texts=_hf_generate_texts,
        batch_size=batch_size,
        collect_samples=collect_samples,
    )
    model.train()
    return out, samples
