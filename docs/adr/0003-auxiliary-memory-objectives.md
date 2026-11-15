# ADR 0003: Add prediction and consistency losses for associative memory

## Status

Accepted.

## Context

The memory branch was trained only through the language-modeling loss. The
original concept also defines an explicit memory objective. Without it, the
delta rule can still fit the next-token objective, but there is no direct
signal that the memory predicts a value before a write or retrieves it after
the write.

## Decision

`AssociativeMemory.forward_with_losses` returns two scalar objectives:

- `prediction`: mean squared error between the value and the read *before* the
  write, encouraging the memory to anticipate new associations;
- `consistency`: mean squared error between the value and the read *after* the
  write, encouraging the write to be retrievable.

The model exposes both as `memory_loss` and `consistency_loss`. The trainer adds
them to the total loss with `memory_loss_coef` and `consistency_loss_coef`
(default `0.0`, so existing configs are unchanged). The 100M config enables
both at `0.01`.

Loss tracking is disabled when both coefficients are zero to avoid the extra
reads. A dedicated ablation of the coefficients is still required.

## Consequences

- Memory has a direct training signal independent of the language head.
- Two extra reads per token when enabled, which is measurable on CPU.
- Coefficients need tuning; the current values are initial, not validated.
