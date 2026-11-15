from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

T = TypeVar("T")


def _build_dataclass(cls: type[T], data: dict[str, Any], name: str) -> T:
    known = {field.name for field in fields(cast(Any, cls))}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "aurora"
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 6

    attention_heads: int = 8
    ffn_hidden: int = 2048

    n_levels: int = 4
    d_state: int = 128

    memory_heads: int = 4
    memory_decay_init: float = 0.9
    memory_write_init: float = 0.5
    memory_epsilon: float = 1e-6

    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter: float = 0.0
    memory_loss_coef: float = 0.0
    consistency_loss_coef: float = 0.0

    dropout: float = 0.0
    norm_eps: float = 1e-6
    layer_scale_init: float = 0.1
    init_std: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.architecture not in {"aurora", "transformer"}:
            raise ValueError("architecture must be one of aurora, transformer")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive")
        if self.attention_heads <= 0 or self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if (self.d_model // self.attention_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive")
        if self.n_levels <= 0 or self.d_state <= 0:
            raise ValueError("n_levels and d_state must be positive")
        if self.memory_heads <= 0:
            raise ValueError("memory_heads must be positive")
        if self.d_model % self.memory_heads != 0:
            raise ValueError("d_model must be divisible by memory_heads")
        if self.n_experts <= 0 or not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if self.expert_hidden <= 0:
            raise ValueError("expert_hidden must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.memory_decay_init < 1.0:
            raise ValueError("memory_decay_init must be in (0, 1)")
        if not 0.0 < self.memory_write_init < 1.0:
            raise ValueError("memory_write_init must be in (0, 1)")
        if self.memory_epsilon <= 0.0:
            raise ValueError("memory_epsilon must be positive")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")
        if (
            self.aux_loss_coef < 0.0
            or self.router_z_loss_coef < 0.0
            or self.memory_loss_coef < 0.0
            or self.consistency_loss_coef < 0.0
        ):
            raise ValueError("Loss coefficients must be non-negative")
        if not 0.0 <= self.router_jitter < 1.0:
            raise ValueError("router_jitter must be in [0, 1)")
        if self.init_std <= 0.0 or self.layer_scale_init <= 0.0:
            raise ValueError("init_std and layer_scale_init must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    train_path: str = ""
    train_paths: tuple[str, ...] = ()
    train_weights: tuple[float, ...] = ()
    val_path: str = ""
    seq_len: int = 512
    dtype: str = "uint16"
    batch_size: int = 4
    grad_accum_steps: int = 8
    num_workers: int = 0

    max_steps: int = 100000
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    eval_interval: int = 500
    eval_batches: int = 20
    log_interval: int = 10
    save_interval: int = 1000
    max_tokens: int = 0
    early_stopping_patience: int = 0
    output_dir: str = "runs/aurora_100m"
    resume: str = ""
    auto_resume: bool = False
    seed: int = 42
    device: str = "auto"
    dtype_amp: str = "auto"
    compile: bool = False
    gradient_checkpointing: bool = False
    tensorboard: bool = False

    qat_enabled: bool = False
    qat_start_step: int = 0
    qat_group_size: int = 128
    qat_min_features: int = 256

    def __post_init__(self) -> None:
        if self.train_paths:
            if self.train_path:
                raise ValueError("train_path and train_paths are mutually exclusive")
            if len(self.train_paths) != len(self.train_weights):
                raise ValueError("train_paths and train_weights must have the same length")
            if any(not path for path in self.train_paths):
                raise ValueError("train_paths must not contain empty paths")
            if any(weight <= 0.0 or not math.isfinite(weight) for weight in self.train_weights):
                raise ValueError("train_weights must be positive and finite")
            object.__setattr__(self, "train_paths", tuple(self.train_paths))
            object.__setattr__(
                self, "train_weights", tuple(float(weight) for weight in self.train_weights)
            )
        elif self.train_weights:
            raise ValueError("train_weights requires train_paths")
        if self.dtype not in {"uint16", "uint32", "int32", "int64"}:
            raise ValueError("dtype must be one of uint16, uint32, int32, int64")
        if self.seq_len <= 0 or self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("seq_len, batch_size and grad_accum_steps must be positive")
        if self.max_steps <= 0 or self.num_workers < 0:
            raise ValueError("max_steps must be positive and num_workers non-negative")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.adam_eps <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("adam_eps and grad_clip must be positive")
        if self.eval_interval < 0 or self.eval_batches < 0 or self.save_interval < 0:
            raise ValueError("eval and save intervals must be non-negative")
        if self.max_tokens < 0 or self.early_stopping_patience < 0:
            raise ValueError("max_tokens and early_stopping_patience must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.dtype_amp not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("dtype_amp must be one of auto, bf16, fp16, fp32")
        if self.qat_group_size <= 0 or self.qat_min_features <= 0:
            raise ValueError("qat_group_size and qat_min_features must be positive")
        if self.qat_start_step < 0:
            raise ValueError("qat_start_step must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        unknown = set(data) - {"model", "training"}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        if "model" not in data or "training" not in data:
            raise ValueError("Config must contain 'model' and 'training' sections")
        model = _build_dataclass(ModelConfig, data["model"], "model")
        training = _build_dataclass(TrainingConfig, data["training"], "training")
        return cls(model=model, training=training)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return ExperimentConfig.from_dict(data)
