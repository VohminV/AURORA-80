from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from aurora.architecture.model import AuroraModel
from aurora.architecture.state import AuroraState
from aurora.inference import inference_autocast, load_model_from_checkpoint
from aurora.tokenizer import load_tokenizer
from aurora.utils import resolve_device


def _apply_repetition_penalty(
    logits: torch.Tensor, token_ids: Sequence[int], penalty: float
) -> torch.Tensor:
    if penalty <= 1.0 or not token_ids:
        return logits
    logits = logits.clone()
    unique_ids = torch.tensor(sorted(set(token_ids)), device=logits.device, dtype=torch.long)
    values = logits[unique_ids]
    logits[unique_ids] = torch.where(values > 0, values / penalty, values * penalty)
    return logits


def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    threshold = torch.topk(logits, top_k).values[-1]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    sorted_logits[cumulative - probabilities > top_p] = float("-inf")
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return filtered


@torch.no_grad()
def generate(
    model: AuroraModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[list[int], AuroraState]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("generate expects input_ids with shape (1, time)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    model.eval()
    output = model(input_ids)
    state = output.state
    logits = output.logits[:, -1, :].float()
    history = input_ids[0].tolist()
    generated: list[int] = []

    for _ in range(max_new_tokens):
        scores = _apply_repetition_penalty(logits[0], history, repetition_penalty)
        if temperature <= 0.0:
            next_token = int(torch.argmax(scores).item())
        else:
            scores = scores / max(temperature, 1e-5)
            scores = _filter_top_k(scores, top_k)
            scores = _filter_top_p(scores, top_p)
            probabilities = F.softmax(scores, dim=-1)
            next_token = int(
                torch.multinomial(probabilities, num_samples=1, generator=generator).item()
            )

        generated.append(next_token)
        history.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break

        step_input = torch.tensor([[next_token]], device=input_ids.device, dtype=torch.long)
        output = model(step_input, state=state)
        state = output.state
        logits = output.logits[:, -1, :].float()

    return generated, state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with an AURORA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", choices=["byte", "sentencepiece"], default="byte")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device=device)

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_model)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    input_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with inference_autocast(device, config.training.dtype_amp):
        generated_ids, _ = generate(
            model,
            input_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_id=tokenizer.eos_id,
            generator=generator,
        )
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
