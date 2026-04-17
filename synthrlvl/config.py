from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from .types import PrefillMode, StepRange, TaskConfig, TemplateName


@dataclass(frozen=True)
class EvalLoopConfig:
    synthetic_step_min: int
    synthetic_step_max: int
    synthetic_samples_per_step: int
    max_new_tokens: int
    generation_backend: str
    generation_batch_size: int
    vllm_gpu_memory_utilization: float
    external_enabled: bool
    external_benchmarks: list[str]
    external_limit: int
    constrained_enabled: bool
    constrained_samples_per_step: int
    constrained_max_lines: int


def task_config_from_cfg(cfg: DictConfig) -> TaskConfig:
    return TaskConfig(
        template=TemplateName(cfg.task.template),
        prefill=PrefillMode(cfg.task.prefill),
        distractor_ratio=float(cfg.task.distractor_ratio),
        train_steps=StepRange(int(cfg.task.train_min_step), int(cfg.task.train_max_step)),
        val_steps=StepRange(int(cfg.task.val_min_step), int(cfg.task.val_max_step)),
        seed=int(cfg.seed),
    )


def eval_loop_config_from_sft(cfg: DictConfig) -> EvalLoopConfig:
    return EvalLoopConfig(
        synthetic_step_min=int(cfg.validation.step_min),
        synthetic_step_max=int(cfg.validation.step_max),
        synthetic_samples_per_step=int(cfg.validation.samples_per_step),
        max_new_tokens=int(cfg.validation.max_new_tokens),
        generation_backend=str(cfg.validation.get("generation_backend", "hf")),
        generation_batch_size=int(cfg.validation.get("generation_batch_size", 16)),
        vllm_gpu_memory_utilization=float(cfg.validation.get("vllm_gpu_memory_utilization", 0.85)),
        external_enabled=bool(cfg.external_eval.enabled),
        external_benchmarks=list(cfg.external_eval.benchmarks),
        external_limit=int(cfg.external_eval.limit_per_benchmark),
        constrained_enabled=bool(cfg.constrained_eval.enabled),
        constrained_samples_per_step=int(cfg.constrained_eval.samples_per_step),
        constrained_max_lines=int(cfg.constrained_eval.max_lines),
    )


def eval_loop_config_from_grpo(cfg: DictConfig) -> EvalLoopConfig:
    return EvalLoopConfig(
        synthetic_step_min=int(cfg.eval.synthetic_step_min),
        synthetic_step_max=int(cfg.eval.synthetic_step_max),
        synthetic_samples_per_step=int(cfg.eval.synthetic_samples_per_step),
        max_new_tokens=int(cfg.eval.max_new_tokens),
        generation_backend=str(cfg.eval.get("generation_backend", "vllm")),
        generation_batch_size=int(cfg.eval.get("generation_batch_size", 1024)),
        vllm_gpu_memory_utilization=float(cfg.eval.get("vllm_gpu_memory_utilization", 0.85)),
        external_enabled=bool(cfg.eval.external_enabled),
        external_benchmarks=list(cfg.eval.external_benchmarks),
        external_limit=int(cfg.eval.external_limit_per_benchmark),
        constrained_enabled=bool(cfg.eval.constrained_enabled),
        constrained_samples_per_step=int(cfg.eval.constrained_samples_per_step),
        constrained_max_lines=int(cfg.eval.constrained_max_lines),
    )
