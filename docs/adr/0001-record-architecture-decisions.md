# ADR 0001: Record architecture decisions

## Status

Accepted.

## Context

AURORA changes quickly: the recurrent core, memory, routing, quantization and
training stack evolve independently. Without a lightweight decision log the
rationale behind previous choices is lost, and experiments become hard to
reproduce.

## Decision

Every decision that changes the architecture, the training protocol, the data
format or the evaluation methodology is recorded as a numbered Markdown file in
`docs/adr/`. Each record contains status, context, decision and consequences.
Records are immutable once accepted; a later record supersedes an earlier one.

## Consequences

- Reviewers can reconstruct why a design exists, not only what it does.
- Superseded decisions remain visible instead of being silently rewritten.
- Small changes still require a short written rationale, which slows down
  accidental complexity.
