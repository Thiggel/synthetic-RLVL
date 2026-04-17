from __future__ import annotations

import os
from typing import Any

import numpy as np
import wandb


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def install_grpo_train_patch(
    *,
    train_gen_every: int | None = None,
    train_gen_num_samples: int | None = None,
    train_gen_max_chars: int | None = None,
    dump_jsonl: bool | None = None,
) -> None:
    if train_gen_every is None:
        train_gen_every = int(os.environ.get("SYNTHRLVL_GRPO_TRAIN_GEN_EVERY", "100") or 100)
    if train_gen_num_samples is None:
        train_gen_num_samples = int(os.environ.get("SYNTHRLVL_GRPO_TRAIN_GEN_NUM_SAMPLES", "4") or 4)
    if train_gen_max_chars is None:
        train_gen_max_chars = int(os.environ.get("SYNTHRLVL_GRPO_TRAIN_GEN_MAX_CHARS", "2000") or 2000)
    if dump_jsonl is None:
        dump_jsonl = os.environ.get("SYNTHRLVL_GRPO_TRAIN_DUMP_JSONL", "0") in {"1", "true", "True"}

    from verl.trainer.ppo import ray_trainer

    RayPPOTrainer = ray_trainer.RayPPOTrainer
    if getattr(RayPPOTrainer, "_syntheval_train_patch_installed", False):
        return

    original_compute_data_metrics = ray_trainer.compute_data_metrics
    original_log_rollout_data = RayPPOTrainer._log_rollout_data

    def patched_compute_data_metrics(batch, use_critic: bool = True) -> dict[str, Any]:
        metrics = original_compute_data_metrics(batch=batch, use_critic=use_critic)
        non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
        for key, raw_values in non_tensor_batch.items():
            if not isinstance(key, str) or not key.startswith("reward/"):
                continue
            vals = _to_float_list(raw_values)
            if not vals:
                continue
            metrics[f"train_aux/{key}/mean"] = float(np.mean(vals))
            metrics[f"train_aux/{key}/min"] = float(np.min(vals))
            metrics[f"train_aux/{key}/max"] = float(np.max(vals))
        return metrics

    def patched_log_rollout_data(self, batch, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str):
        if dump_jsonl:
            original_log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

        if train_gen_every <= 0 or train_gen_num_samples <= 0:
            return
        step = int(getattr(self, "global_steps", 0) or 0)
        if step % train_gen_every != 0:
            return
        if wandb.run is None:
            return

        prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
        gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

        n = min(len(prompts), len(outputs), len(scores), len(gts))
        if n <= 0:
            return

        sample_count = min(int(train_gen_num_samples), n)
        sample_idx = np.random.RandomState(step + 17).choice(n, size=sample_count, replace=False).tolist()

        def _at(name: str, i: int) -> Any:
            vals = reward_extra_infos_dict.get(name)
            if vals is None:
                return None
            if hasattr(vals, "tolist"):
                vals = vals.tolist()
            if not isinstance(vals, list):
                vals = [vals]
            return vals[i] if i < len(vals) else None

        table = wandb.Table(
            columns=[
                "source",
                "step",
                "prompt",
                "generation",
                "gold_answer",
                "score",
                "reward_format",
                "reward_correct",
                "reward_valid",
                "reward_line_match",
            ]
        )
        for i in sample_idx:
            table.add_data(
                "train",
                step,
                str(prompts[i])[: int(train_gen_max_chars)],
                str(outputs[i])[: int(train_gen_max_chars)],
                gts[i],
                float(scores[i]),
                _at("reward/format", i),
                _at("reward/correct", i),
                _at("reward/valid", i),
                _at("reward/line_match", i),
            )
        wandb.log({"train/generations": table}, step=step)

    ray_trainer.compute_data_metrics = patched_compute_data_metrics
    RayPPOTrainer._log_rollout_data = patched_log_rollout_data
    RayPPOTrainer._syntheval_train_patch_installed = True
