from __future__ import annotations

import argparse
import random
from pathlib import Path

ENGLISH = [
    "AURORA combines recurrent state, associative memory and adaptive experts.",
    "The model predicts the next token from the entire preceding context.",
    "A good architecture spends compute only where it is needed.",
    "Associative memory retrieves information by content rather than position.",
    "Experts specialize during training instead of being assigned by hand.",
]

RUSSIAN = [
    "Аврора объединяет рекуррентное состояние, ассоциативную память и экспертов.",
    "Модель предсказывает следующий токен по всему предшествующему контексту.",
    "Хорошая архитектура расходует вычисления только там, где это необходимо.",
    "Ассоциативная память извлекает информацию по содержимому, а не по позиции.",
    "Эксперты специализируются в процессе обучения, а не задаются вручную.",
]

CODE = [
    "def add(a, b):\n    return a + b\n",
    "def square(x):\n    return x * x\n",
    "for i in range(5):\n    print(i)\n",
    "values = [1, 2, 3, 4]\nprint(sum(values))\n",
    "if value > 0:\n    result = value\nelse:\n    result = 0\n",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a small AURORA demo corpus")
    parser.add_argument("--output", default="data/demo.txt")
    parser.add_argument("--lines", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    chunks: list[str] = []

    for _ in range(args.lines):
        kind = rng.randrange(5)
        if kind == 0:
            chunks.append(rng.choice(ENGLISH))
        elif kind == 1:
            chunks.append(rng.choice(RUSSIAN))
        elif kind == 2:
            chunks.append(rng.choice(CODE).strip())
        elif kind == 3:
            left = rng.randint(0, 99)
            right = rng.randint(0, 99)
            operator = rng.choice(["+", "-", "*"])
            result = {"+": left + right, "-": left - right, "*": left * right}[operator]
            chunks.append(f"{left} {operator} {right} = {result}")
        else:
            letters = [rng.choice("ABCDabcd") for _ in range(12)]
            chunks.append(" ".join(letters))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {args.lines} chunks to {output}")


if __name__ == "__main__":
    main()
