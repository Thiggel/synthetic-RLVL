from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import wandb

from .config import EvalLoopConfig
from .types import TaskConfig


def install_grpo_eval_patch(
    task_cfg: TaskConfig,
    eval_cfg: EvalLoopConfig,
    *,
    enabled: bool = True,
    run_verl_validation: bool = False,
) -> None:
    if not enabled:
        return

    from verl.protocol import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.ray_trainer import pad_dataproto_to_divisor, unpad_dataproto

    if getattr(RayPPOTrainer, "_syntheval_patch_installed", False):
        return

    original_validate = RayPPOTrainer._validate
    original_fit = RayPPOTrainer.fit

    def _run_rollout_synthetic_eval(self, *, step: int, collect_samples: int, run_dir: Path):
        from .eval_loop import UnifiedEvaluator

        evaluator = UnifiedEvaluator()
        synthetic_records_by_step = evaluator._build_synthetic_records(task_cfg=task_cfg, eval_cfg=eval_cfg)
        synthetic_records = [
            r
            for s in range(eval_cfg.synthetic_step_min, eval_cfg.synthetic_step_max + 1)
            for r in synthetic_records_by_step[s]
        ]
        first_record_by_prompt: dict[str, object] = {}
        for rec in synthetic_records:
            first_record_by_prompt.setdefault(rec.prompt, rec)

        rollout_cfg = self.config.actor_rollout_ref.rollout
        val_sampling = rollout_cfg.val_kwargs
        response_cap = int(rollout_cfg.response_length)
        batch_size = max(1, int(eval_cfg.generation_batch_size))

        warned_response_cap = bool(getattr(self, "_syntheval_warned_response_cap", False))
        sanity_only = os.environ.get("SYNTHRLVL_VERL_SANITY_ONLY", "0") in {"1", "true", "True"}
        sanity_prompt_path = os.environ.get("SYNTHRLVL_VERL_SANITY_PROMPT_PATH", "").strip()
        sanity_max_new_tokens_env = os.environ.get("SYNTHRLVL_VERL_SANITY_MAX_NEW_TOKENS", "").strip()

        def generate_with_n(
            prompts: list[str],
            max_new_tokens: int,
            do_sample: bool,
            temperature: float,
            num_samples: int,
        ) -> list[str]:
            nonlocal warned_response_cap
            if not prompts:
                return []

            requested_tokens = int(max_new_tokens)
            if requested_tokens > response_cap and not warned_response_cap:
                print(
                    f"[syntheval] requested max_new_tokens={requested_tokens} but rollout.response_length={response_cap}; "
                    "truncating to rollout cap."
                )
                warned_response_cap = True
                self._syntheval_warned_response_cap = True

            orig_temperature = float(val_sampling.temperature)
            orig_top_p = float(val_sampling.top_p)
            orig_top_k = int(val_sampling.top_k)
            orig_do_sample = bool(val_sampling.do_sample)
            orig_n = int(val_sampling.n)
            val_sampling.temperature = float(temperature) if do_sample else 0.0
            val_sampling.top_p = 0.95 if do_sample else 1.0
            val_sampling.top_k = -1
            val_sampling.do_sample = bool(do_sample)
            val_sampling.n = max(1, int(num_samples))

            generations: list[str] = []
            prompt_batch_size = max(1, batch_size // max(1, int(num_samples)))
            num_chunks = (len(prompts) + prompt_batch_size - 1) // prompt_batch_size
            try:
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * prompt_batch_size
                    end = min(len(prompts), start + prompt_batch_size)
                    chunk = prompts[start:end]
                    print(f"[syntheval] generating chunk {chunk_idx + 1}/{num_chunks} ({len(chunk)} prompts) at step={step}")

                    raw_prompt = np.array([[{"role": "user", "content": p}] for p in chunk], dtype=object)
                    chunk_records = [first_record_by_prompt.get(p) for p in chunk]
                    reward_model = []
                    extra_info = []
                    for rec, prompt_text in zip(chunk_records, chunk, strict=True):
                        if rec is None:
                            reward_model.append({"style": "rule", "ground_truth": "", "scorer": "exact_answer"})
                            extra_info.append(
                                {
                                    "question": prompt_text,
                                    "template": task_cfg.template.value,
                                    "prefill": task_cfg.prefill.value,
                                    "schema": "correct_plus_valid_plus_0p1_format",
                                    "gold_logic_premises": "",
                                    "gold_logic_conclusion": "",
                                    "gold_first_modality_lines": [],
                                    "depth": 0,
                                }
                            )
                        else:
                            reward_model.append(
                                {"style": "rule", "ground_truth": rec.gold_answer, "scorer": "exact_answer"}
                            )
                            extra_info.append(
                                {
                                    "question": rec.prompt,
                                    "template": rec.template.value,
                                    "prefill": rec.prefill.value,
                                    "schema": "correct_plus_valid_plus_0p1_format",
                                    "gold_logic_premises": rec.gold_logic_premises,
                                    "gold_logic_conclusion": rec.gold_logic_conclusion,
                                    "gold_first_modality_lines": rec.gold_first_modality_lines,
                                    "depth": int(rec.step),
                                }
                            )

                    batch = DataProto.from_dict(
                        tensors={"dummy_tensor": torch.zeros((len(chunk), 1), dtype=torch.uint8)},
                        non_tensors={
                            "raw_prompt": raw_prompt,
                            "index": np.arange(len(chunk), dtype=np.int64),
                            "data_source": np.array(["synthetic-rlvl"] * len(chunk), dtype=object),
                            "reward_model": np.array(reward_model, dtype=object),
                            "extra_info": np.array(extra_info, dtype=object),
                        },
                    )
                    batch.meta_info = {
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": bool(do_sample),
                        "validate": True,
                        "global_steps": step,
                    }
                    size_divisor = max(1, int(rollout_cfg.agent.num_workers))
                    padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
                    generated_padded = self.async_rollout_manager.generate_sequences(padded)
                    generated = unpad_dataproto(generated_padded, pad_size=pad_size)
                    response_ids = generated.batch["responses"]
                    generations.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids])
            finally:
                val_sampling.temperature = orig_temperature
                val_sampling.top_p = orig_top_p
                val_sampling.top_k = orig_top_k
                val_sampling.do_sample = orig_do_sample
                val_sampling.n = orig_n

            return generations

        def generate_texts(prompts: list[str], max_new_tokens: int, do_sample: bool, temperature: float) -> list[str]:
            return generate_with_n(prompts, max_new_tokens, do_sample, temperature, 1)

        def generate_samples(prompts: list[str], max_new_tokens: int, num_samples: int, temperature: float) -> list[list[str]]:
            n = max(1, int(num_samples))
            flat = generate_with_n(prompts, max_new_tokens, True, temperature, n)
            expected = len(prompts) * n
            if len(flat) != expected:
                raise RuntimeError(f"Sampled VERL eval returned {len(flat)} outputs for {len(prompts)} prompts with n={n}")
            return [flat[i * n : (i + 1) * n] for i in range(len(prompts))]

        def run_sanity_probe() -> None:
            sanity_prompt = ""
            if sanity_prompt_path:
                try:
                    sanity_prompt = Path(sanity_prompt_path).read_text(encoding="utf-8")
                except Exception as exc:
                    print(f"[syntheval] failed to read sanity prompt file {sanity_prompt_path}: {exc}")
            if not sanity_prompt:
                sanity_prompt = synthetic_records[0].prompt if synthetic_records else ""
            if not sanity_prompt:
                print("[syntheval] sanity probe skipped: no prompt available")
                return

            try:
                requested = int(sanity_max_new_tokens_env) if sanity_max_new_tokens_env else min(response_cap, 256)
            except Exception:
                requested = min(response_cap, 256)
            max_new_tokens = max(1, min(int(requested), int(response_cap)))

            # Reuse the rollout engine itself for a single deterministic probe.
            raw_prompt = np.array([[{"role": "user", "content": sanity_prompt}]], dtype=object)
            batch = DataProto.from_dict(
                tensors={"dummy_tensor": torch.zeros((1, 1), dtype=torch.uint8)},
                non_tensors={
                    "raw_prompt": raw_prompt,
                    "index": np.array([0], dtype=np.int64),
                    "data_source": np.array(["synthetic-rlvl"], dtype=object),
                    "reward_model": np.array([{"style": "rule", "ground_truth": "", "scorer": "exact_answer"}], dtype=object),
                    "extra_info": np.array([{"question": sanity_prompt}], dtype=object),
                },
            )
            batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "global_steps": step,
            }
            size_divisor = max(1, int(rollout_cfg.agent.num_workers))
            padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
            generated_padded = self.async_rollout_manager.generate_sequences(padded)
            generated = unpad_dataproto(generated_padded, pad_size=pad_size)
            response_ids_tensor = generated.batch["responses"][0]
            response_ids = [int(x) for x in response_ids_tensor.tolist()]
            decoded_no_skip = self.tokenizer.decode(response_ids_tensor, skip_special_tokens=False)
            decoded_skip = self.tokenizer.decode(response_ids_tensor, skip_special_tokens=True)
            prompt_ids_with_special = self.tokenizer(sanity_prompt, add_special_tokens=True)["input_ids"]
            prompt_ids_no_special = self.tokenizer(sanity_prompt, add_special_tokens=False)["input_ids"]

            payload = {
                "step": int(step),
                "max_new_tokens": int(max_new_tokens),
                "tokenizer_class": self.tokenizer.__class__.__name__,
                "bos_token": self.tokenizer.bos_token,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token": self.tokenizer.eos_token,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token": self.tokenizer.pad_token,
                "pad_token_id": self.tokenizer.pad_token_id,
                "sanity_prompt": sanity_prompt,
                "sanity_prompt_token_ids_with_special": prompt_ids_with_special,
                "sanity_prompt_token_ids_no_special": prompt_ids_no_special,
                "generated_response_token_ids": response_ids,
                "generated_response_decoded_skip_special_tokens_false": decoded_no_skip,
                "generated_response_decoded_skip_special_tokens_true": decoded_skip,
            }

            print("[syntheval] sanity probe payload:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            out_path = run_dir / f"verl_sanity_probe_step{int(step)}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[syntheval] sanity probe written to {out_path}")

        if sanity_only:
            run_sanity_probe()
            return {"syntheval/sanity_probe_done": 1.0}, []

        started = time.time()
        metrics, samples = evaluator._evaluate_with_generator(
            task_cfg=task_cfg,
            eval_cfg=eval_cfg,
            collect_samples=int(collect_samples),
            generate_texts=generate_texts,
            generate_samples=generate_samples,
        )
        elapsed = time.time() - started
        print(f"[syntheval] completed in {elapsed:.1f}s at step={step}")
        return metrics, samples

    def patched_validate(self, merged: bool = False):
        if merged or run_verl_validation:
            result = original_validate(self, merged=merged)
        else:
            result = {"syntheval/placeholder": 0.0}

        step = int(getattr(self, "global_steps", 0) or 0)

        done_steps = getattr(self, "_syntheval_done_steps", None)
        if done_steps is None:
            done_steps = set()
            self._syntheval_done_steps = done_steps
        if step in done_steps:
            return result

        run_dir = Path(str(self.config.trainer.default_local_dir))

        try:
            collect_samples = int(getattr(self.config.trainer, "log_val_generations", 0) or 0)
            metrics, samples = _run_rollout_synthetic_eval(
                self,
                step=step,
                collect_samples=collect_samples,
                run_dir=run_dir,
            )
        except Exception:
            err_path = run_dir / "inprocess_eval_errors.log"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with err_path.open("a", encoding="utf-8") as f:
                f.write(f"step={step}\n")
                f.write(traceback.format_exc())
                f.write("\n")
            if isinstance(result, dict) and result:
                return result
            return {"syntheval/error": 1.0}

        payload = {**metrics, "step": step}
        if wandb.run is not None:
            wandb.log(payload, step=step)
            if samples:
                table = wandb.Table(
                    columns=["source", "step", "prompt", "generation", "gold_answer", "format_ok", "correct", "valid"]
                )
                for row in samples:
                    table.add_data(
                        row["source"],
                        row["step"],
                        str(row["prompt"])[:2000],
                        str(row["generation"])[:2000],
                        row["gold_answer"],
                        row["format_ok"],
                        row["correct"],
                        row["valid"],
                    )
                wandb.log({"val/generations": table}, step=step)

        out_path = run_dir / "inprocess_eval_metrics.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        if os.environ.get("SYNTHRLVL_VERL_SANITY_ONLY", "0") in {"1", "true", "True"}:
            print("[syntheval] sanity-only mode complete; exiting process.")
            os._exit(0)

        done_steps.add(step)
        if not merged and isinstance(result, dict):
            result = dict(result)
            result.update(metrics)
            result["syntheval/step"] = float(step)
        return result

    RayPPOTrainer._validate = patched_validate
    def patched_fit(self):
        result = original_fit(self)
        total_steps = int(getattr(self.config.trainer, "total_training_steps", 0) or 0)
        if total_steps > 0:
            try:
                self.global_steps = total_steps
                patched_validate(self, merged=False)
            except Exception:
                run_dir = Path(str(self.config.trainer.default_local_dir))
                err_path = run_dir / "inprocess_eval_errors.log"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(f"final_step={total_steps}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
        return result

    RayPPOTrainer.fit = patched_fit
    RayPPOTrainer._syntheval_patch_installed = True
