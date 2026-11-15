# ADR 0002: Keep the non-attention recurrent core as the research hypothesis

## Status

Accepted.

## Context

The project hypothesis is that a multi-level recurrent core plus associative
memory and sparse experts can model language without self-attention. Replacing
the core with attention would make the prototype easier to tune, but it would
remove the property the project is trying to test.

## Decision

`AuroraModel` remains the primary architecture. It uses no self-attention:
each block combines a multi-level gated recurrent core, a delta-rule
associative memory and a Top-K MoE branch.

A decoder-only Transformer is added as a *baseline*, not as a replacement. Both
models implement the same output interface so that they can be trained and
evaluated by the same code under equal data, token budget and compute.

## Consequences

- The prototype stays aligned with the stated research question.
- Fair comparison requires a Transformer config with a matched parameter count
  (`configs/transformer_100m.yaml`, 106.28M vs AURORA's 105.73M total).
- AURORA's active parameter count (49.10M) differs from its total parameter
  count because of MoE routing; comparisons must state whether they use total
  or active compute.
