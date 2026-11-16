# Model card: AURORA-100M (prototype draft)

> Research prototype. Do not interpret this card as evidence of quality, safety or superiority over Transformer or any other architecture.

AURORA is a non-Transformer research architecture with a multi-level recurrent core, associative memory and sparse expert routing. This card describes a prototype configuration or checkpoint and must not be read as evidence of quality or superiority over other architectures.

## Model summary

| Field | Value |
| --- | --- |
| Model name | AURORA-100M (prototype draft) |
| Architecture | `aurora` |
| Status | Configuration only (no trained checkpoint) |
| Total parameters | 105,727,560 (105.73M) |
| Active parameters | 49,104,456 (49.10M) |
| Active share | 46.4% |
| License | Not determined |
| Checkpoint | `none` |
| Checkpoint SHA-256 | `n/a` |
| Checkpoint format | n/a |
| Training step | n/a |
| QAT active | n/a |

## Architecture

| Field | Value |
| --- | --- |
| Vocabulary size | 32,768 |
| d_model | 512 |
| Layers | 6 |
| Recurrent levels | 4 |
| d_state | 128 |
| Attention heads | 8 |
| FFN hidden | 2,048 |
| Associative memory | yes |
| Memory heads | 4 |
| MoE enabled | yes |
| Experts per layer | 8 |
| Top-k experts | 2 |
| Expert hidden | 1,024 |
| Dropout | 0 |
| Tied embeddings | yes |
| LayerScale init | 0.1 |
| Memory loss coefficient | 0.01 |
| Consistency loss coefficient | 0.01 |

## Training

| Field | Value |
| --- | --- |
| Train data | `data/processed/train.bin` |
| Validation data | `data/processed/val.bin` |
| Sequence length | 512 |
| Token dtype | `uint16` |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Max steps | 100,000 |
| Max tokens | unlimited |
| Learning rate | 0.0003 |
| Warmup steps | 1,000 |
| Minimum LR ratio | 0.1 |
| Weight decay | 0.1 |
| Gradient clipping | 1 |
| AMP dtype | `auto` |
| Seed | 42 |
| QAT enabled | no |

## Provenance

No checkpoint provenance is available because this card was generated from a configuration only.

## Evaluation

No evaluation report was provided. This card must not be used to claim quality, parity or superiority.

## Intended use

Research on recurrent and associative language models, reproducible ablations, quantization studies and inference-runtime prototyping.

## Out of scope

Production deployment, safety-critical use, medical or legal advice, autonomous decision making, or any claim of parity with existing language models without equal-data, equal-compute evaluations.

## Limitations and risks

- No evaluation results are associated with this card.
- No trained weights are associated with this card.
- The architecture has not demonstrated superiority over Transformer baselines.
- Public benchmark results and a fair equal-compute comparison are not available yet.
- The 80B configuration is a parameter-count target and has not been trained.
- Document-boundary state resets and long-context behavior are still under research.
- Fast ternary kernels and a memory-mapped expert runtime are not implemented.
- Safety alignment, red teaming and data-licensing review are incomplete.
- This card is generated from configuration and run provenance; it does not independently prove data licensing, deduplication quality or benchmark contamination checks.

## Reproducibility

This draft is reproducible from the configuration in the repository. It will be finalized after a trained checkpoint and evaluation report are available.
