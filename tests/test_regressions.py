from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from posttrain_grpo_verl import _rollout_mode_for_backend
from synthrlvl.external_eval import BenchmarkSpec, evaluate_external_benchmarks


class _DummyBatch(dict):
    def to(self, _device: torch.device):
        return self


class _DummyTokenizer:
    eos_token_id = 0

    def __call__(self, _prompt: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        return _DummyBatch({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})

    def decode(self, _ids, skip_special_tokens: bool = True):
        assert skip_special_tokens
        return ""


class _DummyModel:
    def eval(self):
        return self

    def train(self):
        return self

    def generate(self, **kwargs):
        _ = kwargs
        # Return only prompt tokens so decoded continuation is empty.
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def test_external_eval_handles_empty_generation(monkeypatch):
    from synthrlvl import external_eval as mod

    monkeypatch.setattr(
        mod,
        "SPECS",
        {"dummy": BenchmarkSpec("dummy", lambda _limit: [("q", "gold")])},
    )

    metrics, samples = evaluate_external_benchmarks(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        names=["dummy"],
        limit_per_benchmark=1,
        max_new_tokens=8,
        device=torch.device("cpu"),
        collect_samples=1,
    )

    assert "external/dummy/acc" in metrics
    assert metrics["external/dummy/acc"] == 0.0
    assert len(samples) == 1
    assert samples[0]["correct"] == 0.0


def test_verl_reward_importable_as_standalone_module():
    path = Path(__file__).resolve().parents[1] / "synthrlvl" / "verl_reward.py"
    spec = importlib.util.spec_from_file_location("standalone_verl_reward", str(path))
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = module.compute_score(  # type: ignore[attr-defined]
        data_source="synthetic-rlvl",
        solution_str="<answer>yes</answer>",
        ground_truth="yes",
        extra_info={"template": "logic", "prefill": "none", "schema": "correct_plus_valid_plus_0p1_format"},
    )
    assert isinstance(out, dict)
    assert "score" in out


def test_rollout_mode_matches_backend():
    assert _rollout_mode_for_backend("vllm") == "async"
    assert _rollout_mode_for_backend("hf") == "async"
