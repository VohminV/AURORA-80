from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from aurora.architecture.factory import build_model
from aurora.config import load_config
from aurora.utils import human_int, parameter_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count model parameters without allocating weights"
    )
    parser.add_argument("--config", default="configs/aurora_100m.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    with torch.device("meta"):
        model = build_model(config.model)
    total, active = parameter_counts(model)
    ratio = 100.0 * active / max(total, 1)
    print(f"total:  {total:>15,} ({human_int(total)})")
    print(f"active: {active:>15,} ({human_int(active)}, {ratio:.1f}%)")


if __name__ == "__main__":
    main()
