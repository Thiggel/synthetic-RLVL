from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import gc
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from logic_engine import LogicEngine

from .config import EvalLoopConfig
from .evaluation.pass_at_k import score_pass_at_k
from .external_eval import evaluate_external_benchmarks_with_generate_fn
from .generation import ConstrainedProofGenerationConfig, ConstrainedProofGenerator
from .metrics import OutputEvaluator
from .task import TaskBuilder
from .types import StepRange, TaskConfig, TemplateName


@dataclass(frozen=True)
class _EvalPromptRecord:
    step: int
    prompt: str
    gold_answer: str
    template: TemplateName
    prefill: object
    gold_logic_premises: str
    gold_logic_conclusion: str
    gold_first_modality_lines: list[str]


class _HFTextGenerator:
    def __init__(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        batch_size: int,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = max(1, int(batch_size))
        self._pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> list[str]:
        outputs: list[str] = []
        for i in range(0, len(prompts), self._batch_size):
            chunk = list(prompts[i : i + self._batch_size])
            toks = self._tokenizer(chunk, return_tensors="pt", padding=True).to(self._device)
            kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self._pad_token_id,
            )
            if do_sample:
                kwargs["temperature"] = float(temperature)
                kwargs["top_p"] = 0.95
            out_ids = self._model.generate(**toks, **kwargs)
            prompt_lens = toks["attention_mask"].sum(dim=1).tolist()
            for row_idx in range(len(chunk)):
                prompt_len = int(prompt_lens[row_idx])
                text = self._tokenizer.decode(out_ids[row_idx][prompt_len:], skip_special_tokens=True)
                outputs.append(text)
        return outputs

    @torch.no_grad()
    def generate_many(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        num_samples: int,
        temperature: float,
    ) -> list[list[str]]:
        n = max(1, int(num_samples))
        if n == 1:
            return [[text] for text in self.generate(prompts, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)]

        grouped: list[list[str]] = []
        prompt_batch_size = max(1, self._batch_size // n)
        for i in range(0, len(prompts), prompt_batch_size):
            chunk = list(prompts[i : i + prompt_batch_size])
            toks = self._tokenizer(chunk, return_tensors="pt", padding=True).to(self._device)
            out_ids = self._model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                num_return_sequences=n,
                pad_token_id=self._pad_token_id,
            )
            prompt_lens = toks["attention_mask"].sum(dim=1).tolist()
            for row_idx in range(len(chunk)):
                samples: list[str] = []
                prompt_len = int(prompt_lens[row_idx])
                for sample_idx in range(n):
                    out_idx = row_idx * n + sample_idx
                    text = self._tokenizer.decode(out_ids[out_idx][prompt_len:], skip_special_tokens=True)
                    samples.append(text)
                grouped.append(samples)
        return grouped


class _VLLMTextGenerator:
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str,
        batch_size: int,
        gpu_memory_utilization: float,
        lora_adapter_path: str | None,
        lora_base_model_name: str | None,
    ):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self._batch_size = max(1, int(batch_size))
        self._llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=False,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_num_seqs=max(1, int(batch_size)),
            enable_lora=bool(lora_adapter_path),
            disable_log_stats=True,
        )
        self._lora_request = (
            LoRARequest(
                lora_name="syntheval",
                lora_int_id=1,
                lora_path=str(lora_adapter_path),
                base_model_name=lora_base_model_name,
            )
            if lora_adapter_path is not None
            else None
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> list[str]:
        from vllm import SamplingParams

        sampling = SamplingParams(
            n=1,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature) if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
        )
        outputs: list[str] = []
        for i in range(0, len(prompts), self._batch_size):
            chunk = list(prompts[i : i + self._batch_size])
            req_outputs = self._llm.generate(
                chunk,
                sampling_params=sampling,
                use_tqdm=False,
                lora_request=self._lora_request,
            )
            for req in req_outputs:
                text = req.outputs[0].text if req.outputs else ""
                outputs.append(text)
        return outputs

    def generate_many(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        num_samples: int,
        temperature: float,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        n = max(1, int(num_samples))
        if self._lora_request is None:
            sampling = SamplingParams(
                n=n,
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=0.95,
            )
            grouped: list[list[str]] = []
            prompt_batch_size = max(1, self._batch_size // n)
            for i in range(0, len(prompts), prompt_batch_size):
                chunk = list(prompts[i : i + prompt_batch_size])
                print(
                    f"[syntheval] sampled vLLM chunk {i // prompt_batch_size + 1}/"
                    f"{(len(prompts) + prompt_batch_size - 1) // prompt_batch_size} "
                    f"({len(chunk)} prompts x n={n})",
                    flush=True,
                )
                req_outputs = self._llm.generate(
                    chunk,
                    sampling_params=sampling,
                    use_tqdm=False,
                    lora_request=None,
                )
                for req in req_outputs:
                    grouped.append([out.text for out in req.outputs])
            return grouped

        sampling = SamplingParams(
            n=1,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=0.95,
        )
        grouped: list[list[str]] = [[] for _ in prompts]
        expanded: list[tuple[int, str]] = [
            (prompt_idx, prompt)
            for prompt_idx, prompt in enumerate(prompts)
            for _ in range(n)
        ]
        for i in range(0, len(expanded), self._batch_size):
            chunk_pairs = expanded[i : i + self._batch_size]
            chunk = [prompt for _, prompt in chunk_pairs]
            print(
                f"[syntheval] sampled vLLM chunk {i // self._batch_size + 1}/"
                f"{(len(expanded) + self._batch_size - 1) // self._batch_size} "
                f"({len(chunk)} sequences, n={n})",
                flush=True,
            )
            req_outputs = self._llm.generate(
                chunk,
                sampling_params=sampling,
                use_tqdm=False,
                lora_request=self._lora_request,
            )
            for (prompt_idx, _), req in zip(chunk_pairs, req_outputs, strict=True):
                text = req.outputs[0].text if req.outputs else ""
                grouped[prompt_idx].append(text)
        return grouped


class UnifiedEvaluator:
    def __init__(self):
        self.output_eval = OutputEvaluator()
        self.engine = LogicEngine()

    def _build_synthetic_records(
        self,
        *,
        task_cfg: TaskConfig,
        eval_cfg: EvalLoopConfig,
    ) -> dict[int, list[_EvalPromptRecord]]:
        by_step: dict[int, list[_EvalPromptRecord]] = {}
        for step in range(eval_cfg.synthetic_step_min, eval_cfg.synthetic_step_max + 1):
            step_cfg = replace(task_cfg, val_steps=StepRange(step, step))
            rows = TaskBuilder(step_cfg).build_samples(eval_cfg.synthetic_samples_per_step, train=False)
            by_step[step] = [
                _EvalPromptRecord(
                    step=step,
                    prompt=s.prompt,
                    gold_answer=s.answer,
                    template=step_cfg.template,
                    prefill=step_cfg.prefill,
                    gold_logic_premises=s.logic_premises,
                    gold_logic_conclusion=s.logic_conclusion,
                    gold_first_modality_lines=s.gold_first_modality_lines,
                )
                for s in rows
            ]
        return by_step

    def _score_synthetic_outputs(
        self,
        *,
        records: Sequence[_EvalPromptRecord],
        generations: Sequence[str],
        eval_cfg: EvalLoopConfig,
        collect_samples: int,
    ) -> tuple[Dict[str, float], List[dict]]:
        if len(records) != len(generations):
            raise RuntimeError(f"Mismatched synthetic eval lengths: {len(records)} records vs {len(generations)} generations")

        metrics: Dict[str, float] = {}
        sample_candidates_by_step: dict[int, list[dict]] = defaultdict(list)
        vals_by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"syntactic": [], "format": [], "correct": [], "valid": []})
        for rec, gen in zip(records, generations, strict=True):
            score = self.output_eval.evaluate(
                gen,
                template=rec.template,
                gold_answer=rec.gold_answer,
                gold_logic_premises=rec.gold_logic_premises,
                gold_logic_conclusion=rec.gold_logic_conclusion,
                prefill=rec.prefill,
                gold_first_modality_lines=rec.gold_first_modality_lines,
            )
            vals_by_step[rec.step]["syntactic"].append(score.syntactic)
            vals_by_step[rec.step]["format"].append(score.format_ok)
            vals_by_step[rec.step]["correct"].append(score.correct)
            vals_by_step[rec.step]["valid"].append(score.valid)
            if collect_samples > 0:
                sample_candidates_by_step[rec.step].append(
                    {
                        "source": "synthetic",
                        "step": rec.step,
                        "prompt": rec.prompt,
                        "generation": gen,
                        "gold_answer": rec.gold_answer,
                        "syntactic": score.syntactic,
                        "format_ok": score.format_ok,
                        "correct": score.correct,
                        "valid": score.valid,
                    }
                )

        for step in range(eval_cfg.synthetic_step_min, eval_cfg.synthetic_step_max + 1):
            vals = vals_by_step.get(step, {"syntactic": [], "format": [], "correct": [], "valid": []})
            for key in ("syntactic", "format", "correct", "valid"):
                metrics[f"synthetic/step_{step}/{key}"] = sum(vals[key]) / max(1, len(vals[key]))

        samples: List[dict] = []
        if collect_samples > 0:
            steps = list(range(eval_cfg.synthetic_step_min, eval_cfg.synthetic_step_max + 1))
            offsets = {step: 0 for step in steps}
            while len(samples) < collect_samples:
                added = False
                for step in steps:
                    candidates = sample_candidates_by_step.get(step, [])
                    offset = offsets[step]
                    if offset >= len(candidates):
                        continue
                    samples.append(candidates[offset])
                    offsets[step] = offset + 1
                    added = True
                    if len(samples) >= collect_samples:
                        break
                if not added:
                    break
        return metrics, samples

    def _evaluate_with_generator(
        self,
        *,
        task_cfg: TaskConfig,
        eval_cfg: EvalLoopConfig,
        collect_samples: int,
        generate_texts: Callable[[Sequence[str], int, bool, float], list[str]],
        generate_samples: Callable[[Sequence[str], int, int, float], list[list[str]]] | None = None,
    ) -> tuple[Dict[str, float], List[dict]]:
        records_by_step = self._build_synthetic_records(task_cfg=task_cfg, eval_cfg=eval_cfg)
        records = [r for step in range(eval_cfg.synthetic_step_min, eval_cfg.synthetic_step_max + 1) for r in records_by_step[step]]
        generations = generate_texts([r.prompt for r in records], eval_cfg.max_new_tokens, False, 0.0)
        metrics, samples = self._score_synthetic_outputs(
            records=records,
            generations=generations,
            eval_cfg=eval_cfg,
            collect_samples=collect_samples,
        )

        if eval_cfg.sampled_enabled:
            sample_n = max(1, int(eval_cfg.sampled_num_generations))
            k_values = [int(k) for k in eval_cfg.sampled_k_values if int(k) <= sample_n]
            if k_values:
                prompts = [r.prompt for r in records]
                if generate_samples is None:
                    sampled_generations = self._generate_samples_by_repetition(
                        prompts=prompts,
                        num_samples=sample_n,
                        max_new_tokens=eval_cfg.max_new_tokens,
                        temperature=eval_cfg.sampled_temperature,
                        generate_texts=generate_texts,
                    )
                else:
                    sampled_generations = generate_samples(
                        prompts,
                        eval_cfg.max_new_tokens,
                        sample_n,
                        eval_cfg.sampled_temperature,
                    )
                train_max = int(task_cfg.train_steps.max_step)
                band_predicates = {
                    "train": lambda step, train_max=train_max: step <= train_max,
                    "ood": lambda step, train_max=train_max: step > train_max,
                    "hard_tail": lambda step: step >= 15,
                }
                metrics.update(
                    score_pass_at_k(
                        records=records,
                        generations_by_record=sampled_generations,
                        output_eval=self.output_eval,
                        k_values=k_values,
                        band_predicates=band_predicates,
                    )
                )

        if eval_cfg.constrained_enabled and task_cfg.template == TemplateName.LOGIC:
            constrained_records = [
                rec
                for step in range(eval_cfg.synthetic_step_min, eval_cfg.synthetic_step_max + 1)
                for rec in records_by_step[step][: max(1, int(eval_cfg.constrained_samples_per_step))]
            ]
            if constrained_records:
                constrained_many = generate_samples
                if constrained_many is None:
                    constrained_many = lambda prompts, max_tokens, num_samples, temperature: self._generate_samples_by_repetition(
                        prompts=prompts,
                        num_samples=num_samples,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        generate_texts=generate_texts,
                    )
                constrained = ConstrainedProofGenerator(generate_many=constrained_many)
                constrained_config = ConstrainedProofGenerationConfig(
                    num_generations=max(1, int(eval_cfg.constrained_num_generations)),
                    candidates_per_line=max(1, int(eval_cfg.constrained_candidates_per_line)),
                    max_lines=max(1, int(eval_cfg.constrained_max_lines)),
                    max_line_tokens=max(1, int(eval_cfg.constrained_max_line_tokens)),
                    suffix_max_tokens=max(1, int(eval_cfg.constrained_suffix_max_tokens)),
                    temperature=float(eval_cfg.constrained_temperature),
                )
                constrained_generations, constrained_traces = constrained.generate_many(
                    [r.prompt for r in constrained_records],
                    max_new_tokens=eval_cfg.max_new_tokens,
                    config=constrained_config,
                )
                constrained_k = [
                    int(k)
                    for k in eval_cfg.constrained_k_values
                    if 1 <= int(k) <= constrained_config.num_generations
                ]
                if constrained_k:
                    train_max = int(task_cfg.train_steps.max_step)
                    metrics.update(
                        score_pass_at_k(
                            records=constrained_records,
                            generations_by_record=constrained_generations,
                            output_eval=self.output_eval,
                            k_values=constrained_k,
                            metric_prefix="synthetic_constrained_sampled",
                            band_predicates={
                                "train": lambda step, train_max=train_max: step <= train_max,
                                "ood": lambda step, train_max=train_max: step > train_max,
                                "hard_tail": lambda step: step >= 15,
                            },
                        )
                    )
                for rec, generations_for_rec, traces_for_rec in zip(
                    constrained_records,
                    constrained_generations,
                    constrained_traces,
                    strict=True,
                ):
                    if len(samples) >= collect_samples:
                        break
                    if not generations_for_rec:
                        continue
                    score = self.output_eval.evaluate(
                        generations_for_rec[0],
                        template=rec.template,
                        gold_answer=rec.gold_answer,
                        gold_logic_premises=rec.gold_logic_premises,
                        gold_logic_conclusion=rec.gold_logic_conclusion,
                        prefill=rec.prefill,
                        gold_first_modality_lines=rec.gold_first_modality_lines,
                    )
                    trace = traces_for_rec[0] if traces_for_rec else None
                    samples.append(
                        {
                            "source": "synthetic_constrained",
                            "step": rec.step,
                            "prompt": rec.prompt,
                            "generation": generations_for_rec[0],
                            "gold_answer": rec.gold_answer,
                            "syntactic": score.syntactic,
                            "format_ok": score.format_ok,
                            "correct": score.correct,
                            "valid": score.valid,
                            "constrained_trace": None
                            if trace is None
                            else {
                                "used_constrained_proof": trace.used_constrained_proof,
                                "proof_lines": trace.proof_lines,
                                "candidate_calls": trace.candidate_calls,
                                "best_scores": list(trace.best_scores),
                            },
                        }
                    )

        if eval_cfg.external_enabled:
            ext_metrics, ext_samples = evaluate_external_benchmarks_with_generate_fn(
                names=eval_cfg.external_benchmarks,
                limit_per_benchmark=eval_cfg.external_limit,
                max_new_tokens=eval_cfg.max_new_tokens,
                generate_texts=lambda prompts, max_tokens: generate_texts(prompts, max_tokens, False, 0.0),
                batch_size=eval_cfg.generation_batch_size,
                collect_samples=max(0, collect_samples - len(samples)),
            )
            metrics.update(ext_metrics)
            samples.extend(ext_samples)

        return metrics, samples

    def _generate_samples_by_repetition(
        self,
        *,
        prompts: Sequence[str],
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        generate_texts: Callable[[Sequence[str], int, bool, float], list[str]],
    ) -> list[list[str]]:
        grouped: list[list[str]] = [[] for _ in prompts]
        for _ in range(max(1, int(num_samples))):
            generations = generate_texts(prompts, max_new_tokens, True, temperature)
            if len(generations) != len(prompts):
                raise RuntimeError(f"Sampled generation returned {len(generations)} outputs for {len(prompts)} prompts")
            for idx, gen in enumerate(generations):
                grouped[idx].append(gen)
        return grouped

    @torch.no_grad()
    def evaluate_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        task_cfg: TaskConfig,
        eval_cfg: EvalLoopConfig,
        device: torch.device,
        collect_samples: int = 0,
    ) -> tuple[Dict[str, float], List[dict]]:
        model.eval()
        try:
            generator = _HFTextGenerator(
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=eval_cfg.generation_batch_size,
            )
            return self._evaluate_with_generator(
                task_cfg=task_cfg,
                eval_cfg=eval_cfg,
                collect_samples=collect_samples,
                generate_texts=lambda prompts, max_tokens, do_sample, temperature: generator.generate(
                    prompts,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                ),
                generate_samples=lambda prompts, max_tokens, num_samples, temperature: generator.generate_many(
                    prompts,
                    max_new_tokens=max_tokens,
                    num_samples=num_samples,
                    temperature=temperature,
                ),
            )
        finally:
            model.train()

    def _resolve_checkpoint_paths(self, checkpoint_dir: str | Path) -> tuple[str, str, Path | None]:
        raw_checkpoint = str(checkpoint_dir)
        checkpoint_path = Path(raw_checkpoint).expanduser()
        adapter_dir: Path | None = None
        tokenizer_path: str = raw_checkpoint
        model_path: str = raw_checkpoint

        # VERL actor checkpoints store tokenizer/config under `huggingface/`
        # and LoRA weights under `lora_adapter/`.
        if checkpoint_path.exists():
            checkpoint_path = checkpoint_path.resolve()
            tokenizer_path = str(checkpoint_path)
            model_path = str(checkpoint_path)
        if checkpoint_path.is_dir():
            actor_hf = checkpoint_path / "huggingface"
            actor_lora = checkpoint_path / "lora_adapter"
            if actor_hf.is_dir():
                tokenizer_path = str(actor_hf)
                model_path = str(actor_hf)
            if actor_lora.is_dir() and (actor_lora / "adapter_config.json").is_file():
                adapter_dir = actor_lora
            elif (checkpoint_path / "adapter_config.json").is_file() and (checkpoint_path / "adapter_model.safetensors").is_file():
                adapter_dir = checkpoint_path
        return model_path, tokenizer_path, adapter_dir

    def _evaluate_checkpoint_hf(
        self,
        *,
        model_path: str,
        tokenizer_path: str,
        adapter_dir: Path | None,
        collect_samples: int,
        task_cfg: TaskConfig,
        eval_cfg: EvalLoopConfig,
    ) -> tuple[Dict[str, float], List[dict]]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if adapter_dir is not None:
            payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
            base_model = payload.get("base_model_name_or_path")
            if not isinstance(base_model, str) or not base_model.strip():
                raise ValueError(f"Adapter checkpoint at {adapter_dir} is missing base_model_name_or_path")
            base = AutoModelForCausalLM.from_pretrained(base_model.strip(), torch_dtype=torch.bfloat16, device_map="auto")
            model = PeftModel.from_pretrained(base, str(adapter_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        try:
            return self.evaluate_model(
                model,
                tokenizer,
                device=model.device,
                collect_samples=collect_samples,
                task_cfg=task_cfg,
                eval_cfg=eval_cfg,
            )
        finally:
            del model
            torch.cuda.empty_cache()

    def _evaluate_checkpoint_vllm(
        self,
        *,
        model_path: str,
        tokenizer_path: str,
        adapter_dir: Path | None,
        collect_samples: int,
        task_cfg: TaskConfig,
        eval_cfg: EvalLoopConfig,
    ) -> tuple[Dict[str, float], List[dict]]:
        lora_adapter_path: str | None = None
        lora_base_model_name: str | None = None
        vllm_model_path = model_path
        if adapter_dir is not None:
            payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
            base_model = payload.get("base_model_name_or_path")
            if not isinstance(base_model, str) or not base_model.strip():
                raise ValueError(f"Adapter checkpoint at {adapter_dir} is missing base_model_name_or_path")
            vllm_model_path = base_model.strip()
            lora_adapter_path = str(adapter_dir)
            lora_base_model_name = base_model.strip()

        generator = _VLLMTextGenerator(
            model_path=vllm_model_path,
            tokenizer_path=tokenizer_path,
            batch_size=eval_cfg.generation_batch_size,
            gpu_memory_utilization=eval_cfg.vllm_gpu_memory_utilization,
            lora_adapter_path=lora_adapter_path,
            lora_base_model_name=lora_base_model_name,
        )
        try:
            return self._evaluate_with_generator(
                task_cfg=task_cfg,
                eval_cfg=eval_cfg,
                collect_samples=collect_samples,
                generate_texts=lambda prompts, max_tokens, do_sample, temperature: generator.generate(
                    prompts,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                ),
                generate_samples=lambda prompts, max_tokens, num_samples, temperature: generator.generate_many(
                    prompts,
                    max_new_tokens=max_tokens,
                    num_samples=num_samples,
                    temperature=temperature,
                ),
            )
        finally:
            del generator
            gc.collect()
            torch.cuda.empty_cache()

    def evaluate_checkpoint(
        self,
        checkpoint_dir: str | Path,
        collect_samples: int = 0,
        **kwargs,
    ) -> tuple[Dict[str, float], List[dict]]:
        task_cfg: TaskConfig = kwargs["task_cfg"]
        eval_cfg: EvalLoopConfig = kwargs["eval_cfg"]
        model_path, tokenizer_path, adapter_dir = self._resolve_checkpoint_paths(checkpoint_dir)
        backend = str(eval_cfg.generation_backend).strip().lower()
        if backend not in {"auto", "hf", "vllm"}:
            raise ValueError(f"Unsupported eval generation backend: {eval_cfg.generation_backend}")

        if backend in {"auto", "vllm"}:
            try:
                return self._evaluate_checkpoint_vllm(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    adapter_dir=adapter_dir,
                    collect_samples=collect_samples,
                    task_cfg=task_cfg,
                    eval_cfg=eval_cfg,
                )
            except Exception:
                if backend == "vllm":
                    raise

        return self._evaluate_checkpoint_hf(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            adapter_dir=adapter_dir,
            collect_samples=collect_samples,
            task_cfg=task_cfg,
            eval_cfg=eval_cfg,
        )
