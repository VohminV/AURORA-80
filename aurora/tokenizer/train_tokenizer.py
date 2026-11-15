from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an AURORA SentencePiece tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Text files for training")
    parser.add_argument("--model-prefix", required=True, help="Output model prefix")
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--model-type", choices=["bpe", "unigram", "char"], default="bpe")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--no-byte-fallback",
        action="store_true",
        help="Disable SentencePiece byte fallback",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise SystemExit("Install sentencepiece to train a tokenizer") from error

    for input_path in args.input:
        if not Path(input_path).is_file():
            raise SystemExit(f"Input file not found: {input_path}")

    output_prefix = Path(args.model_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=str(output_prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=not args.no_byte_fallback,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        num_threads=args.num_threads,
    )
    print(f"Tokenizer written to {output_prefix}.model")


if __name__ == "__main__":
    main()
