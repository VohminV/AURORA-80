from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from aurora.architecture.factory import (
    CausalLanguageModel,
    build_model,
    set_gradient_checkpointing,
)
from aurora.config import ExperimentConfig
from aurora.data.provenance import (
    DatasetSource,
    collect_dataset_sources,
    describe_dataset_sources,
)
from aurora.provenance import build_run_metadata, utc_now
from aurora.quantization import convert_to_ternary
from aurora.training.checkpoint import load_checkpoint, save_checkpoint
from aurora.training.optimizer import create_optimizer
from aurora.training.scheduler import create_scheduler
from aurora.training.tracker import WandbTracker
from aurora.utils import (
    human_int,
    load_torch_rng_state_dict,
    parameter_counts,
    resolve_amp_dtype,
    resolve_device,
    seed_worker,
    set_seed,
    torch_rng_state_dict,
)


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        train_sampler: Sampler[int] | None = None,
        dataset_sources: Sequence[DatasetSource] | None = None,
    ) -> None:
        self.config = config
        self.training = config.training
        self.logger = logging.getLogger("aurora.trainer")
        self.device = resolve_device(self.training.device)
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        set_seed(self.training.seed)

        self.model = build_model(config.model).to(self.device)
        if self.training.gradient_checkpointing:
            blocks = set_gradient_checkpointing(self.model, True)
            self.logger.info("Enabled gradient checkpointing for %d blocks", blocks)
        total_parameters, active_parameters = parameter_counts(self.model)
        self.logger.info(
            "Model: %s total parameters, %s active parameters",
            human_int(total_parameters),
            human_int(active_parameters),
        )

        self.qat_active = False
        self.optimizer = create_optimizer(self.model, self.training, self.device)
        self.scheduler = create_scheduler(self.optimizer, self.training)
        self.amp_dtype = resolve_amp_dtype(self.device, self.training.dtype_amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True) if self.amp_dtype == torch.float16 else None
        )

        self.train_loader = self._build_loader(train_dataset, shuffle=True, sampler=train_sampler)
        self.val_loader = (
            self._build_loader(val_dataset, shuffle=False) if val_dataset is not None else None
        )

        self.output_dir = Path(self.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if dataset_sources is None:
            configured_train_paths = (
                self.training.train_paths
                if self.training.train_paths
                else ((self.training.train_path,) if self.training.train_path else ())
            )
            dataset_sources = collect_dataset_sources(
                configured_train_paths,
                self.training.val_path or None,
                train_weights=self.training.train_weights,
                manifest_paths=self.training.dataset_manifests,
                verify_files=self.training.verify_data,
                expected_dtype=self.training.dtype,
            )
        self.dataset_sources = list(dataset_sources)
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.run_metadata = build_run_metadata(self.device)
        self.run_metadata["dataset"] = describe_dataset_sources(self.dataset_sources)
        self.run_metadata["resume_count"] = 0
        self.last_train_metrics: dict[str, Any] | None = None
        self.last_validation_metrics: dict[str, Any] | None = None
        self.step = 0
        self.best_val = float("inf")
        self.tokens_seen = 0
        self._evals_without_improvement = 0
        self.train_model: torch.nn.Module = self.model
        self.tracker: WandbTracker | None = None
        self.writer: Any | None = None
        if self.training.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError:
                self.logger.warning("tensorboard is not installed; disabling event logging")

        self._last_log_time = time.perf_counter()
        self._tokens_since_log = 0

    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _record_metrics(self, record: dict[str, Any]) -> None:
        payload = {
            "created_utc": utc_now(),
            "run_id": self.run_metadata.get("run_id"),
            **record,
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _metrics_summary(self) -> dict[str, Any]:
        best_val = self.best_val if math.isfinite(self.best_val) else None
        return {
            "best_val": best_val,
            "tokens_seen": self.tokens_seen,
            "last_train": self.last_train_metrics,
            "last_validation": self.last_validation_metrics,
            "metrics_file": str(self.metrics_file),
        }

    def _build_loader(
        self,
        dataset: Dataset,
        shuffle: bool,
        sampler: Sampler[int] | None = None,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.training.seed)
        dataset_length = len(cast(Any, dataset))
        if sampler is not None:
            return DataLoader(
                dataset,
                batch_size=self.training.batch_size,
                sampler=sampler,
                drop_last=dataset_length >= self.training.batch_size,
                num_workers=self.training.num_workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.training.num_workers > 0,
                worker_init_fn=seed_worker,
                generator=generator,
            )
        return DataLoader(
            dataset,
            batch_size=self.training.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and dataset_length >= self.training.batch_size,
            num_workers=self.training.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.training.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _enable_qat(self) -> None:
        converted = convert_to_ternary(
            self.model,
            group_size=self.training.qat_group_size,
            min_features=self.training.qat_min_features,
        )
        self.qat_active = True
        self.logger.info("Enabled ternary QAT for %d linear layers", converted)

    def _maybe_enable_qat(self, step: int) -> bool:
        if not self.training.qat_enabled or self.qat_active:
            return False
        if step < self.training.qat_start_step:
            return False
        self._enable_qat()
        return True

    def _set_train_model(self) -> None:
        if self.training.compile:
            self.train_model = cast(torch.nn.Module, torch.compile(self.model))
        else:
            self.train_model = self.model

    def _resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, map_location="cpu")
        saved_run = checkpoint.get("run_metadata")
        saved_dataset = saved_run.get("dataset") if isinstance(saved_run, dict) else None
        current_dataset = self.run_metadata["dataset"]
        if isinstance(saved_dataset, dict):
            saved_fingerprint = saved_dataset.get("fingerprint")
            current_fingerprint = current_dataset.get("fingerprint")
            saved_versions = saved_dataset.get("versions")
            versions_changed = bool(saved_versions) and saved_versions != current_dataset.get(
                "versions"
            )
            if (saved_fingerprint is not None and saved_fingerprint != current_fingerprint) or (
                saved_fingerprint is None and versions_changed
            ):
                raise ValueError(
                    "Dataset provenance mismatch: checkpoint was trained on "
                    f"{saved_fingerprint or saved_versions}, current data is "
                    f"{current_fingerprint or current_dataset.get('versions')}. "
                    "Restore the original dataset or start a new run."
                )
        else:
            self.logger.warning(
                "Checkpoint has no dataset provenance; skipping dataset verification"
            )

        saved_model_config = checkpoint.get("config", {}).get("model")
        if saved_model_config is not None and saved_model_config != self.config.to_dict()["model"]:
            self.logger.warning("Checkpoint model config differs from the current config")

        if checkpoint.get("qat_active", False):
            saved_training = checkpoint.get("config", {}).get("training", {})
            for field in ("qat_group_size", "qat_min_features"):
                saved_value = saved_training.get(field)
                if saved_value is not None and saved_value != getattr(self.training, field):
                    raise ValueError(
                        f"Cannot resume a QAT checkpoint with a different {field}: "
                        f"{saved_value} != {getattr(self.training, field)}"
                    )
            if not self.qat_active:
                self._enable_qat()

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
        best_val = checkpoint.get("best_val")
        self.best_val = float(best_val) if best_val is not None else float("inf")
        saved_metrics = checkpoint.get("metrics")
        if isinstance(saved_metrics, dict):
            last_train = saved_metrics.get("last_train")
            if isinstance(last_train, dict):
                self.last_train_metrics = last_train
            last_validation = saved_metrics.get("last_validation")
            if isinstance(last_validation, dict):
                self.last_validation_metrics = last_validation
        if isinstance(saved_run, dict):
            saved_run_id = saved_run.get("run_id")
            if isinstance(saved_run_id, str) and saved_run_id:
                self.run_metadata["run_id"] = saved_run_id
            started_utc = saved_run.get("started_utc")
            if isinstance(started_utc, str):
                self.run_metadata["started_utc"] = started_utc
            original_git = saved_run.get("original_git")
            if not isinstance(original_git, dict):
                original_git = saved_run.get("git")
            if isinstance(original_git, dict):
                self.run_metadata["original_git"] = original_git
            original_environment = saved_run.get("original_environment")
            if not isinstance(original_environment, dict):
                original_environment = saved_run.get("environment")
            if isinstance(original_environment, dict):
                self.run_metadata["original_environment"] = original_environment
            resume_count = saved_run.get("resume_count")
            self.run_metadata["resume_count"] = (
                resume_count + 1 if isinstance(resume_count, int) else 1
            )
        self.run_metadata["resumed_from"] = str(path)
        self.run_metadata["resumed_utc"] = utc_now()
        if checkpoint.get("rng_state"):
            load_torch_rng_state_dict(checkpoint["rng_state"])
        self._record_metrics(
            {
                "type": "resume",
                "step": self.step,
                "resumed_from": str(path),
            }
        )
        self.logger.info("Resumed from %s at step %d", path, self.step)

    def _save(self, name: str) -> None:
        save_checkpoint(
            self.output_dir / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.step,
            self.config,
            self.qat_active,
            best_val=self.best_val,
            rng_state=torch_rng_state_dict(),
            run_metadata=self.run_metadata,
            metrics=self._metrics_summary(),
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with self._autocast():
                logits = cast(CausalLanguageModel, self.model)(inputs).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.model.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )
            total_loss += float(loss)
            total_tokens += targets.numel()
            batches += 1
            if self.training.eval_batches > 0 and batches >= self.training.eval_batches:
                break
        self.model.train()
        return total_loss / max(total_tokens, 1)

    def _evaluate_and_log(self) -> bool:
        val_loss = self.evaluate()
        perplexity = math.exp(min(val_loss, 20.0))
        improved = val_loss < self.best_val
        best_val = val_loss if improved else self.best_val
        self.logger.info(
            "validation step %d | loss %.4f | perplexity %.2f",
            self.step,
            val_loss,
            perplexity,
        )
        if self.writer is not None:
            self.writer.add_scalar("validation/loss", val_loss, self.step)
            self.writer.add_scalar("validation/perplexity", perplexity, self.step)
        if self.tracker is not None:
            self.tracker.log(
                {"validation/loss": val_loss, "validation/perplexity": perplexity},
                step=self.step,
            )
        self.last_validation_metrics = {
            "loss": val_loss,
            "perplexity": perplexity,
            "best_val": best_val,
            "improved": improved,
        }
        self._record_metrics(
            {
                "type": "validation",
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                **self.last_validation_metrics,
            }
        )
        if improved:
            self.best_val = val_loss
            self._evals_without_improvement = 0
            self._save("best.pt")
            self.logger.info("New best validation loss: %.4f", val_loss)
        else:
            self._evals_without_improvement += 1

        patience = self.training.early_stopping_patience
        return patience > 0 and self._evals_without_improvement >= patience

    def _log_step(
        self,
        lm_loss: float,
        aux_loss: float,
        grad_norm: float,
        memory_loss: float = 0.0,
        consistency_loss: float = 0.0,
    ) -> None:
        elapsed = time.perf_counter() - self._last_log_time
        tokens_per_second = self._tokens_since_log / max(elapsed, 1e-9)
        learning_rate = float(self.scheduler.get_last_lr()[0])
        tokens_seen = self.step * self.training.batch_size * self.training.grad_accum_steps
        tokens_seen *= self.training.seq_len
        self.logger.info(
            "step %d/%d | lm %.4f | aux %.4f | mem %.4f | cons %.4f | lr %.2e | "
            "grad %.3f | %.0f tok/s | %.2fM tokens",
            self.step,
            self.training.max_steps,
            lm_loss,
            aux_loss,
            memory_loss,
            consistency_loss,
            learning_rate,
            grad_norm,
            tokens_per_second,
            tokens_seen / 1_000_000,
        )
        train_metrics = {
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "memory_loss": memory_loss,
            "consistency_loss": consistency_loss,
            "lr": learning_rate,
            "grad_norm": grad_norm,
            "tokens_per_second": tokens_per_second,
        }
        if self.writer is not None:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, self.step)
        if self.tracker is not None:
            self.tracker.log(
                {f"train/{name}": value for name, value in train_metrics.items()},
                step=self.step,
            )
        self.last_train_metrics = {
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "memory_loss": memory_loss,
            "consistency_loss": consistency_loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "tokens_per_second": tokens_per_second,
        }
        self._record_metrics(
            {
                "type": "train",
                "step": self.step,
                "tokens_seen": tokens_seen,
                **self.last_train_metrics,
            }
        )
        self._last_log_time = time.perf_counter()
        self._tokens_since_log = 0

    def _resolve_resume_path(self) -> str | Path | None:
        if self.training.resume:
            return self.training.resume
        if self.training.auto_resume:
            candidate = self.output_dir / "latest.pt"
            if candidate.is_file():
                return candidate
        return None

    def fit(self) -> None:
        resume_path = self._resolve_resume_path()
        if resume_path is not None:
            self._resume(resume_path)
        self.tracker = WandbTracker.start(self.config, self.output_dir, self.run_metadata)
        self._maybe_enable_qat(self.step)
        self._set_train_model()
        self.model.train()

        train_iterator = iter(self.train_loader)
        tokens_per_step = (
            self.training.batch_size * self.training.grad_accum_steps * self.training.seq_len
        )
        self.tokens_seen = self.step * tokens_per_step
        self.logger.info(
            "Starting training at step %d with %s tokens per optimizer step",
            self.step,
            human_int(tokens_per_step),
        )

        try:
            for step in range(self.step, self.training.max_steps):
                if self._maybe_enable_qat(step):
                    self._set_train_model()
                    self.model.train()

                self.optimizer.zero_grad(set_to_none=True)
                step_finite = True
                lm_loss_sum = 0.0
                aux_loss_sum = 0.0
                memory_loss_sum = 0.0
                consistency_loss_sum = 0.0

                for _ in range(self.training.grad_accum_steps):
                    try:
                        inputs, targets = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(self.train_loader)
                        inputs, targets = next(train_iterator)
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    with self._autocast():
                        output = cast(CausalLanguageModel, self.train_model)(inputs)
                        lm_loss = F.cross_entropy(
                            output.logits.reshape(-1, self.config.model.vocab_size),
                            targets.reshape(-1),
                        )
                        loss = lm_loss + output.aux_loss
                        if self.config.model.memory_loss_coef > 0.0:
                            loss = loss + self.config.model.memory_loss_coef * output.memory_loss
                        if self.config.model.consistency_loss_coef > 0.0:
                            loss = (
                                loss
                                + self.config.model.consistency_loss_coef * output.consistency_loss
                            )

                    if not bool(torch.isfinite(loss)):
                        step_finite = False
                    scaled_loss = loss / self.training.grad_accum_steps
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    lm_loss_sum += float(lm_loss.detach())
                    aux_loss_sum += float(output.aux_loss.detach())
                    memory_loss_sum += float(output.memory_loss.detach())
                    consistency_loss_sum += float(output.consistency_loss.detach())

                if not step_finite:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler is not None:
                        self.scaler.update()
                    self.step = step + 1
                    self.tokens_seen = self.step * tokens_per_step
                    self.logger.warning(
                        "Skipping step %d because the loss is not finite", self.step
                    )
                    continue

                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.training.grad_clip
                    )
                    if not bool(torch.isfinite(grad_norm)):
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                        self.step = step + 1
                        self.tokens_seen = self.step * tokens_per_step
                        self.logger.warning(
                            "Skipping step %d because the gradient norm is not finite",
                            self.step,
                        )
                        continue
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.training.grad_clip
                    )
                    if not bool(torch.isfinite(grad_norm)):
                        self.optimizer.zero_grad(set_to_none=True)
                        self.step = step + 1
                        self.tokens_seen = self.step * tokens_per_step
                        self.logger.warning(
                            "Skipping step %d because the gradient norm is not finite",
                            self.step,
                        )
                        continue
                    self.optimizer.step()

                self.scheduler.step()
                self.step = step + 1
                self.tokens_seen = self.step * tokens_per_step
                self._tokens_since_log += tokens_per_step

                stop_requested = (
                    self.training.max_tokens > 0 and self.tokens_seen >= self.training.max_tokens
                )

                if self.step % self.training.log_interval == 0:
                    self._log_step(
                        lm_loss=lm_loss_sum / self.training.grad_accum_steps,
                        aux_loss=aux_loss_sum / self.training.grad_accum_steps,
                        grad_norm=float(grad_norm),
                        memory_loss=memory_loss_sum / self.training.grad_accum_steps,
                        consistency_loss=consistency_loss_sum / self.training.grad_accum_steps,
                    )

                should_evaluate = (
                    self.val_loader is not None
                    and self.training.eval_interval > 0
                    and self.step % self.training.eval_interval == 0
                )
                if should_evaluate:
                    if self._evaluate_and_log():
                        self.logger.info(
                            "Early stopping at step %d after %d evaluations without improvement",
                            self.step,
                            self._evals_without_improvement,
                        )
                        stop_requested = True

                if self.training.save_interval > 0 and self.step % self.training.save_interval == 0:
                    self._save("latest.pt")

                if stop_requested:
                    if (
                        self.training.max_tokens > 0
                        and self.tokens_seen >= self.training.max_tokens
                    ):
                        self.logger.info(
                            "Stopping at step %d: token budget %d reached (%d tokens seen)",
                            self.step,
                            self.training.max_tokens,
                            self.tokens_seen,
                        )
                    break

            if (
                self.val_loader is not None
                and self.step > 0
                and (
                    self.training.eval_interval <= 0 or self.step % self.training.eval_interval != 0
                )
            ):
                self._evaluate_and_log()

        except KeyboardInterrupt:
            self.logger.info("Training interrupted; saving latest checkpoint")
            self._save("latest.pt")
            raise
        finally:
            if self.tracker is not None:
                self.tracker.finish()
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

        self._save("latest.pt")
