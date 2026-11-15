from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from aurora.config import ExperimentConfig, load_config
from aurora.data.dataset import MemmapTokenDataset
from aurora.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain an AURORA model")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
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
    ):
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value
    if args.tensorboard:
        overrides["tensorboard"] = True
    if args.no_compile:
        overrides["compile"] = False
    if overrides:
        config = ExperimentConfig(
            model=config.model,
            training=replace(config.training, **overrides),
        )

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

    logger.info("Training samples: %d", len(train_dataset))
    if val_dataset is not None:
        logger.info("Validation samples: %d", len(val_dataset))

    trainer = Trainer(config, train_dataset, val_dataset)
    trainer.fit()


if __name__ == "__main__":
    main()
