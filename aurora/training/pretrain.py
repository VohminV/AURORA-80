from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from aurora.config import ExperimentConfig, load_config
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.mixture import build_token_mixture
from aurora.data.provenance import collect_dataset_sources
from aurora.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain an AURORA model")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        default=[],
        dest="dataset_manifests",
        metavar="PATH",
        help="Prepared dataset manifest; repeat to describe mixture sources",
    )
    parser.add_argument("--no-verify-data", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("aurora.pretrain")

    config = load_config(args.config)
    overrides = {}
    for field_name in (
        "train_path",
        "val_path",
        "output_dir",
        "resume",
        "max_steps",
        "device",
        "seed",
        "max_tokens",
        "early_stopping_patience",
    ):
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value
    if args.tensorboard:
        overrides["tensorboard"] = True
    if args.auto_resume:
        overrides["auto_resume"] = True
    if args.no_compile:
        overrides["compile"] = False
    if args.gradient_checkpointing:
        overrides["gradient_checkpointing"] = True
    if args.dataset_manifests:
        overrides["dataset_manifests"] = tuple(args.dataset_manifests)
    if args.no_verify_data:
        overrides["verify_data"] = False
    if overrides:
        config = ExperimentConfig(
            model=config.model,
            training=replace(config.training, **overrides),
        )

    configured_train_paths = (
        config.training.train_paths
        if config.training.train_paths
        else ((config.training.train_path,) if config.training.train_path else ())
    )
    try:
        dataset_sources = collect_dataset_sources(
            configured_train_paths,
            config.training.val_path or None,
            train_weights=config.training.train_weights,
            manifest_paths=config.training.dataset_manifests,
            verify_files=config.training.verify_data,
            expected_dtype=config.training.dtype,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for dataset_source in dataset_sources:
        logger.info(
            "Dataset %s | split %s | manifest version %s | sha256 %s | verified %s",
            dataset_source.path,
            dataset_source.split or "none",
            dataset_source.dataset_version or "unversioned",
            dataset_source.sha256,
            dataset_source.verified,
        )

    train_sampler = None
    if config.training.train_paths:
        try:
            train_dataset, train_sampler, sources = build_token_mixture(
                config.training.train_paths,
                config.training.train_weights,
                config.training.seq_len,
                dtype=config.training.dtype,
                seed=config.training.seed,
            )
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        total_weight = sum(source.weight for source in sources)
        for mixture_source in sources:
            logger.info(
                "Mixture source %s | weight %.3f | samples %d",
                mixture_source.path,
                mixture_source.weight / total_weight,
                mixture_source.samples,
            )
    else:
        train_path = Path(config.training.train_path)
        if not train_path.is_file():
            raise SystemExit(
                f"Training token file not found: {train_path}. "
                "Run aurora.data.prepare first or pass --train-path."
            )
        train_dataset = MemmapTokenDataset(
            train_path, config.training.seq_len, dtype=config.training.dtype
        )

    val_dataset = None
    if config.training.val_path:
        val_path = Path(config.training.val_path)
        if not val_path.is_file():
            raise SystemExit(f"Validation token file not found: {val_path}")
        val_dataset = MemmapTokenDataset(
            val_path, config.training.seq_len, dtype=config.training.dtype
        )

    logger.info("Training samples: %d", len(cast(Any, train_dataset)))
    if val_dataset is not None:
        logger.info("Validation samples: %d", len(val_dataset))

    trainer = Trainer(
        config,
        train_dataset,
        val_dataset,
        train_sampler=train_sampler,
        dataset_sources=dataset_sources,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
