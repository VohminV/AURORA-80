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
evaluation and document-aware packing). This is implemented with an optional
boolean reset mask threaded through `AuroraModel`, `AuroraBlock`,
`RecurrentCore` and `AssociativeMemory`. A `True` entry zeroes the batch state
before that token is processed. The trainer derives the mask from EOS markers:
`mask[:, 0]` is always false and `mask[:, i] = input_ids[:, i - 1] == eos`.

Enable the behaviour with `training.document_reset: true` and
`training.eos_token_id` (default 2). The aurora configs and smoke config enable
it; datasets prepared by `prepare.py` contain an EOS after each document. The
trainer rejects document resets for the Transformer baseline, whose KV cache has
no reset-mask support. Evaluation, perplexity and quantization benchmarks derive
the same mask from the checkpoint config. `AuroraModel.forward` remains backward
compatible: callers that pass no mask keep the old flat-window behaviour.

The reset happens before the first token of the new document. The EOS token
itself is still processed with the previous document's state, and the loss for
predicting the first token of the new document from that EOS position is not
masked; this is a packing/loss-masking decision, not part of the reset
semantics.

## Consequences

- The state contract stays backward compatible; reset is opt-in per config.
- Cross-document state contamination is removed from subsequent tokens in
  packed training sequences when EOS markers are present.
- New documents get a zero initial state, matching the per-sequence contract.
- The Transformer baseline and checkpoints with `document_reset: false` keep
  their previous behaviour.
- `aurora-state-check --reset-interval` remains the reference tool for measuring
  the effect of resets independently of document markers.
