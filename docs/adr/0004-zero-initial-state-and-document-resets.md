# ADR 0004: Zero initial state and document-boundary resets

## Status

Accepted.

## Context

The recurrent core and associative memory are stateful. `AuroraModel.forward`
creates a zero state when the caller passes `None`, and the trainer calls the
model without a state for every batch. Within a batch, state flows across the
whole packed sequence; there is no state reset at document boundaries even
though `prepare.py` appends EOS after each document when the tokenizer has an
EOS id.

Long-context behaviour therefore depends on two separate questions:

- whether a learned (non-zero) initial state is worth its extra parameters;
- whether state must be reset when a new document starts.

`aurora-state-check` already measures both a single continued state and
periodic resets. On the smoke model, resetting every 4,096 tokens changed
neither the 32k loss nor the final state norm within printed precision, but that
model is undertrained and does not answer the quality question.

## Decision

Keep the zero initial state. A learned initial state is not added now: it adds
parameters, complicates checkpoint compatibility and has no supporting evidence
yet. Revisit it during the M2 architecture ablations.

Reset state at document boundaries whenever the boundary is known (inference,
evaluation and any future document-aware packing). Do not implement boundary
resets inside the recurrent kernels yet: current datasets expose only flat token
windows, so a reset mask would have to be threaded through the core, memory and
trainer. This is deferred to the M2 long-context work, where document-aware
packing and the 2k/4k/8k evaluation are built together.

Until then, training uses zero state per sequence and EOS markers as the only
document signal. Evaluation must state whether it continues one state or resets
periodically.

## Consequences

- The state contract stays simple and backward compatible.
- Cross-document contamination is possible inside packed training sequences;
  this is a known M1 limitation, not a silent assumption.
- `aurora-state-check --reset-interval` is the reference way to measure the
  effect of resets before document-aware training exists.
