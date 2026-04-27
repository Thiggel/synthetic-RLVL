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
    sampled_enabled: bool
    sampled_num_generations: int
    sampled_k_values: list[int]
    sampled_temperature: float
    external_enabled: bool
    external_benchmarks: list[str]
    external_limit: int
    constrained_enabled: bool
    constrained_samples_per_step: int
    constrained_max_lines: int
    constrained_num_generations: int
    constrained_candidates_per_line: int
    constrained_max_line_tokens: int
    constrained_suffix_max_tokens: int
    constrained_temperature: float
    constrained_k_values: list[int]


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
        sampled_enabled=bool(cfg.validation.get("sampled_enabled", True)),
        sampled_num_generations=int(cfg.validation.get("sampled_num_generations", 64)),
        sampled_k_values=[int(k) for k in list(cfg.validation.get("sampled_k_values", [1, 2, 4, 8, 16, 32, 64]))],
        sampled_temperature=float(cfg.validation.get("sampled_temperature", 1.0)),
        external_enabled=bool(cfg.external_eval.enabled),
        external_benchmarks=list(cfg.external_eval.benchmarks),
        external_limit=int(cfg.external_eval.limit_per_benchmark),
        constrained_enabled=bool(cfg.get("constrained_eval", {}).get("enabled", False)),
        constrained_samples_per_step=int(cfg.get("constrained_eval", {}).get("samples_per_step", cfg.validation.samples_per_step)),
        constrained_max_lines=int(cfg.get("constrained_eval", {}).get("max_lines", 32)),
        constrained_num_generations=int(cfg.get("constrained_eval", {}).get("num_generations", 8)),
        constrained_candidates_per_line=int(cfg.get("constrained_eval", {}).get("candidates_per_line", 8)),
        constrained_max_line_tokens=int(cfg.get("constrained_eval", {}).get("max_line_tokens", 48)),
        constrained_suffix_max_tokens=int(cfg.get("constrained_eval", {}).get("suffix_max_tokens", 256)),
        constrained_temperature=float(cfg.get("constrained_eval", {}).get("temperature", 1.0)),
        constrained_k_values=[int(k) for k in list(cfg.get("constrained_eval", {}).get("k_values", [1, 2, 4, 8]))],
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
        sampled_enabled=bool(cfg.eval.get("sampled_enabled", True)),
        sampled_num_generations=int(cfg.eval.get("sampled_num_generations", 64)),
        sampled_k_values=[int(k) for k in list(cfg.eval.get("sampled_k_values", [1, 2, 4, 8, 16, 32, 64]))],
        sampled_temperature=float(cfg.eval.get("sampled_temperature", 1.0)),
        external_enabled=bool(cfg.eval.external_enabled),
        external_benchmarks=list(cfg.eval.external_benchmarks),
        external_limit=int(cfg.eval.external_limit_per_benchmark),
        constrained_enabled=bool(cfg.eval.constrained_enabled),
        constrained_samples_per_step=int(cfg.eval.constrained_samples_per_step),
        constrained_max_lines=int(cfg.eval.constrained_max_lines),
        constrained_num_generations=int(cfg.eval.get("constrained_num_generations", 8)),
        constrained_candidates_per_line=int(cfg.eval.get("constrained_candidates_per_line", 8)),
        constrained_max_line_tokens=int(cfg.eval.get("constrained_max_line_tokens", 48)),
        constrained_suffix_max_tokens=int(cfg.eval.get("constrained_suffix_max_tokens", 256)),
        constrained_temperature=float(cfg.eval.get("constrained_temperature", 1.0)),
        constrained_k_values=[int(k) for k in list(cfg.eval.get("constrained_k_values", [1, 2, 4, 8]))],
    )
