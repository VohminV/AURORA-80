from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aurora.architecture.factory import (
    CausalLanguageModel,
    build_model,
    set_gradient_checkpointing,
)
from aurora.config import ExperimentConfig
from aurora.quantization import convert_to_ternary
from aurora.training.checkpoint import load_checkpoint, save_checkpoint
from aurora.training.optimizer import create_optimizer
from aurora.training.scheduler import create_scheduler
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

        self.train_loader = self._build_loader(train_dataset, shuffle=True)
        self.val_loader = (
            self._build_loader(val_dataset, shuffle=False) if val_dataset is not None else None
        )

        self.output_dir = Path(self.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.best_val = float("inf")
        self.tokens_seen = 0
        self._evals_without_improvement = 0
        self.train_model: torch.nn.Module = self.model
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

    def _build_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.training.seed)
        dataset_length = len(cast(Any, dataset))
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
        if checkpoint.get("rng_state"):
            load_torch_rng_state_dict(checkpoint["rng_state"])
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
        self.logger.info(
            "validation step %d | loss %.4f | perplexity %.2f",
            self.step,
            val_loss,
            perplexity,
        )
        if self.writer is not None:
            self.writer.add_scalar("validation/loss", val_loss, self.step)
            self.writer.add_scalar("validation/perplexity", perplexity, self.step)
        if val_loss < self.best_val:
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
        learning_rate = self.scheduler.get_last_lr()[0]
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
        if self.writer is not None:
            self.writer.add_scalar("train/lm_loss", lm_loss, self.step)
            self.writer.add_scalar("train/aux_loss", aux_loss, self.step)
            self.writer.add_scalar("train/memory_loss", memory_loss, self.step)
            self.writer.add_scalar("train/consistency_loss", consistency_loss, self.step)
            self.writer.add_scalar("train/lr", learning_rate, self.step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.step)
            self.writer.add_scalar("train/tokens_per_second", tokens_per_second, self.step)
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
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

        self._save("latest.pt")
