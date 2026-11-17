from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aurora.data.dataset import MemmapTokenDataset
from aurora.inference import (
    inference_autocast,
    load_model_from_checkpoint,
    model_output,
    reset_mask_for_config,
)
from aurora.utils import resolve_device


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate AURORA perplexity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Tokenized .bin file")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=0, help="0 means the full file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device=device)
    seq_len = args.seq_len or config.training.seq_len

    dataset = MemmapTokenDataset(args.data, seq_len, dtype=config.training.dtype)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    batches = 0
    with torch.no_grad(), inference_autocast(device, config.training.dtype_amp):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model_output(
                model, inputs, reset_mask=reset_mask_for_config(inputs, config)
            ).logits
            loss = F.cross_entropy(
                logits.reshape(-1, config.model.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss)
            total_tokens += targets.numel()
            batches += 1
            if args.batches > 0 and batches >= args.batches:
                break

    mean_loss = total_loss / max(total_tokens, 1)
    print(f"loss: {mean_loss:.6f}")
    print(f"perplexity: {math.exp(min(mean_loss, 20.0)):.4f}")
    print(f"tokens: {total_tokens}")


if __name__ == "__main__":
    main()
