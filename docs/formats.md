# Checkpoint and AURORA-TQ formats

## Checkpoint format (`*.pt`)

Checkpoints are written with `torch.save` and loaded with
`torch.load(..., weights_only=True)`. Loading therefore never unpickles arbitrary
Python objects.

Top-level keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `format_version` | `int` | Checkpoint schema version (currently `2`) |
| `model` | `dict[str, Tensor]` | Model `state_dict` |
| `optimizer` | `dict` | AdamW state |
| `scheduler` | `dict` | Cosine scheduler state |
| `scaler` | `dict \| None` | FP16 `GradScaler` state, `None` outside FP16 CUDA |
| `step` | `int` | Number of completed optimizer steps |
| `config` | `dict` | Full experiment config (`model` and `training` sections) |
| `qat_active` | `bool` | Whether ternary QAT conversion is active |
| `best_val` | `float \| None` | Best validation loss seen so far |
| `rng_state` | `dict \| None` | Torch CPU/CUDA RNG state for resume |
| `run_metadata` | `dict` | Run identity, git revision, environment and dataset provenance |
| `metrics` | `dict` | Last train/validation metrics and metrics log path |

Properties:

- Writes are atomic: data is saved to `<name>.tmp` and moved with `os.replace`.
- `model`, `optimizer`, `scheduler`, `scaler`, `step`, `config`, `qat_active` and
  `best_val` are required to resume. `rng_state` is optional.
- Version `2` additionally requires `run_metadata` and `metrics`. Version `1`
  checkpoints still load, but resume logs a warning because they have no dataset
  provenance.
- Old checkpoints without newer optional config fields still load because missing
  `ModelConfig`/`TrainingConfig` fields fall back to dataclass defaults.
- QAT checkpoints replace selected `nn.Linear` modules with `TernaryLinear`. The
  state-dict keys then look like `...linear.weight` instead of `...weight`; resume
  and inference rebuild the same module structure before loading.
- Loading validates required keys, the format version and all model tensors for
  NaN/Inf. Corrupted checkpoints raise `ValueError` instead of failing later in
  the training loop.
- `auto_resume=True` in `TrainingConfig` (CLI: `--auto-resume`) loads
  `<output_dir>/latest.pt` when no explicit `resume` path is given. Explicit
  resume takes precedence.
- There is no encryption, compression or sharding yet. Sharded checkpoints are a
  requirement for the 1B/80B stages.

### Run metadata

`run_metadata` is built when a `Trainer` starts and preserved across resumes:

| Key | Meaning |
| --- | --- |
| `run_id` | Stable id shared by every checkpoint and metrics row of the run |
| `started_utc` | Original run start time (preserved on resume) |
| `resumed_from` | Path of the checkpoint used for the latest resume |
| `resumed_utc` | Time of the latest resume |
| `resume_count` | Number of resumes |
| `git` | `available`, `commit`, `branch`, `dirty`, `dirty_files`, `root` |
| `original_git` | Git state recorded when the run first started (after a resume) |
| `environment` | Python, platform, Torch, NumPy, CUDA, cuDNN, threads, device and GPU info |
| `original_environment` | Environment recorded when the run first started (after a resume) |
| `dataset` | Dataset provenance described below |

`dataset` contains:

| Key | Meaning |
| --- | --- |
| `version` | Common `manifest.dataset_version` when every source shares one |
| `versions` | Sorted unique manifest dataset versions |
| `fingerprint` | SHA-256 over source roles, mixture weights, split, file checksums and dataset versions |
| `verified` | Whether every manifest-backed file checksum was re-verified |
| `manifest_linked` | Whether every source file was found in a manifest |
| `sources` | Per-file role, path, bytes, SHA-256, manifest path, split, tokens, dtype and weight |

Resume rejects a checkpoint when the current dataset fingerprint differs from
the stored one. This catches edited token files, regenerated splits, changed
mixture weights and different dataset versions.

### Metrics log (`metrics.jsonl`)

Each checkpoint save also refers to an append-only `<output_dir>/metrics.jsonl`.
Every row is JSON with `run_id`, `created_utc`, `type` and `step`:

| Type | Extra fields |
| --- | --- |
| `train` | `tokens_seen`, `lm_loss`, `aux_loss`, `memory_loss`, `consistency_loss`, `learning_rate`, `grad_norm`, `tokens_per_second` |
| `validation` | `tokens_seen`, `loss`, `perplexity`, `best_val`, `improved` |
| `resume` | `resumed_from` |

The log is append-only and may contain several `run_id` values if the same
`output_dir` is reused for independent runs. The checkpoint `metrics` entry
stores `best_val`, `tokens_seen`, `last_train`, `last_validation` and the log
path.

## Curation report (`curation_report.json`)

`aurora.data.curate` writes a plain-text corpus plus this JSON report:

| Key | Meaning |
| --- | --- |
| `format_version` | Curation report schema version (currently `1`) |
| `created_utc` | Creation timestamp |
| `config` | Every quality, dedup, PII and normalization setting |
| `inputs` | Per-file path, detected format, bytes, SHA-256, records, `invalid_records` and `truncated` |
| `output` | Path, bytes, SHA-256 and kept document count |
| `counts` | Input/kept counters, script distribution, per-reason drop counters, `dropped_total` and `drop_reasons` |
| `dataset_version` | Content-addressed `sha256:...` version of the curated output |
| `run_metadata` | Git/environment block shared with training and evaluation |
| `rejects` | Optional path to the rejection JSONL |

`dataset_version` hashes the curation format version, the full curation config,
input sizes and SHA-256 values, and the output size and SHA-256. Paths and
timestamps are excluded, so re-running the same curation on the same inputs with
the same filters produces the same version.

Rejected JSONL rows contain `record`, `source`, optional `line` (JSONL input),
`reason`, `characters`, detected `pii` kinds and `preview`. `preview` is `null`
when `pii_action="drop"`; with `redact` or `keep` it stores the first 200
characters (after redaction when configured). The curated output separates
documents with a blank line and can be passed to tokenizer training or
`aurora.data.prepare`.

## Corpus stats report (`corpus_stats.json`)

`aurora-corpus-stats` streams a text/JSONL corpus and writes:

| Key | Meaning |
| --- | --- |
| `format_version` | Report schema version (currently `1`) |
| `created_utc` | Creation timestamp |
| `settings` | Input format, JSON/domain fields, chunk size, encoding, document limit and code-line threshold |
| `inputs` | Per-file path, resolved format and document count |
| `domains` | Per-domain documents, characters, words, lines, code-line fraction, script counts/fractions and `language_hint` |
| `summary` | Aggregated counters, script distribution and overall `language_hint` |
| `limitations` | Heuristic caveats |

The domain label comes from `--domain-field` for JSONL records and falls back
to the file stem. Script ratios are computed per alphabetic Unicode character
and are not a language identification model. `language_hint` reports
`code`/`cyrillic`/`cjk`/`latin`/`mixed`/`unknown` using script fractions and the
share of lines that look like source code. Token counts are intentionally absent
because they depend on the tokenizer; use `aurora-tokenizer-stats` for
fertility.

## Tokenized dataset manifest (`manifest.json`)

## Contamination report (`contamination.json`)

`aurora-contamination` scans a corpus for word n-grams that also occur in
benchmark documents:

```bash
python -m aurora.data.contamination \
  --corpus data/curated/corpus.txt \
  --benchmark benchmarks/mmlu.jsonl benchmarks/hellaswag.txt \
  --ngram-size 13 --output reports/contamination.json --fail-on-match
```

| Key | Meaning |
| --- | --- |
| `format_version` | Report schema version (currently `1`) |
| `created_utc` | Creation timestamp |
| `settings` | N-gram size, input format, JSON field, chunk size, encoding and example limit |
| `benchmarks` | Per-file documents, indexed documents, n-grams, matched documents and contamination fraction |
| `corpus` | Per-file documents, words, n-grams, contaminated documents and matched n-grams |
| `summary` | Aggregate benchmark/corpus counts, matched n-grams and contamination fraction |
| `examples` | Up to `--max-examples` overlaps with corpus path/document/offset/line and benchmark document references |
| `limitations` | Heuristic and collision caveats |

Only benchmark n-gram hashes are kept in memory (64-bit polynomial hashes), so
the corpus is streamed and memory usage scales with the benchmarks rather than
the corpus. Text inputs are split on blank lines; JSONL inputs use
`--json-field`. `--format` forces `text` or `jsonl` for every input. The check is
word-based and case-folded after NFKC normalization; it does not detect
translation, paraphrasing or tokenization-aware contamination. `--fail-on-match`
exits with status 1 after writing the report.

## Tokenized dataset manifest (`manifest.json`)

`aurora.data.prepare` writes a manifest next to `train.bin` and `val.bin`:

| Key | Meaning |
| --- | --- |
| `format_version` | Manifest schema version (currently `1`) |
| `created_utc` | Creation timestamp |
| `dataset_version` | Content-addressed `sha256:...` version of the prepared dataset |
| `tokenizer` | Kind, model path, model SHA-256/bytes and vocabulary size |
| `encoding`, `dtype`, `split_mode`, `val_fraction`, `test_fraction`, `chunk_size_bytes` | Preparation settings |
| `inputs` | Per-file path, byte size, SHA-256 and token count |
| `outputs` | Per-split (`train`, `val`, optional `test`) file name, byte size, SHA-256 and token count |

The `.bin` files are raw little-endian integer arrays (`uint16` or `uint32`),
without a header. Their shape is `(num_tokens,)` and the dtype is recorded in the
manifest and the training config.

`dataset_version` hashes the input/output hashes, token counts, split settings
and tokenizer identity (kind, vocabulary size and model SHA-256), excluding paths
and timestamps. Re-running preparation on the same content therefore produces
the same version. `aurora.data.version.read_manifest` recomputes the version and
rejects edited metadata; `verify_manifest_files` rechecks split sizes and
checksums. `aurora-data-card` renders a Markdown data card from a manifest:

```bash
python -m aurora.data.card --manifest data/processed/manifest.json \
  --output data/processed/DATA_CARD.md --name "AURORA corpus" --verify
```

`split_mode="document"` splits on blank lines, appends EOS after every document
and assigns whole documents to one split. `split_mode="token"` keeps the original
byte-offset split and may cut inside a document.

## Model card (`MODEL_CARD.md`)

`aurora-model-card` renders a Markdown model card from a checkpoint or a config:

```bash
python -m aurora.model_card --checkpoint runs/aurora_100m/best.pt \
  --evaluation reports/aurora_100m_eval.json \
  --name "AURORA-100M" --license "TBD" \
  --output reports/aurora_100m_MODEL_CARD.md

python -m aurora.model_card --config configs/aurora_100m.yaml \
  --name "AURORA-100M (prototype draft)" \
  --output docs/model_card_100m.md
```

With `--checkpoint`, the card reads `format_version`, model config, step,
`qat_active`, `best_val`, metrics and `run_metadata` from the checkpoint, and
includes the checkpoint SHA-256. The checkpoint is read with
`torch.load(..., weights_only=True)` and is not instantiated as a model, so
generating a card does not require matching hardware. Parameter counts (total and
active) are computed from the config on the `meta` device.

With `--evaluation`, the card adds the report's per-split tokens, loss,
perplexity, `dataset_version`, git commit and evaluation settings. Without a
checkpoint the card is explicitly marked `Configuration only (no trained
checkpoint)`; without an evaluation report it states that no quality claim can
be made. `--config` and `--checkpoint` are mutually exclusive. Additional
`--limitation` entries are appended to the standard limitations list.

The generator records provenance and configured data paths, but does not verify
data licenses, deduplication quality, contamination or safety properties. Those
checks must be documented separately before a release.

### Dataset provenance checks

Training and evaluation link every `.bin` file to its manifest:

- A `manifest.json` next to the token file is discovered automatically.
- `TrainingConfig.dataset_manifests` (CLI: repeat `--dataset-manifest PATH`)
  supplies explicit manifests. Every configured train/val path must be described
  by one of them; a file described by multiple manifests is rejected.
- `read_manifest` recomputes `dataset_version` before the file is accepted.
- The matched output's byte size and SHA-256 are checked against the file. The
  manifest dtype must match `TrainingConfig.dtype`.
- `verify_data=False` (CLI: `--no-verify-data`) skips rehashing manifest-backed
  files for faster startup, but still uses the checksum recorded in the
  validated manifest. Unversioned raw token files are always content-hashed.
- Missing manifests are allowed for backwards compatibility and are recorded as
  `dataset_version=None`; use explicit manifests for reproducible runs.
- A fresh run records the resulting dataset fingerprint in its checkpoints. A
  resume fails with `Dataset provenance mismatch` if the data content, mixture
  weights, split assignment or manifest version changed.

## AURORA-TQ (ternary weights)

Implemented in `aurora/quantization/ternary.py`, `qat.py` and `tq.py`.

- Weights are quantized along the last dimension in groups of `group_size`
  (default 128; the last group is padded).
- For each group, `scale = mean(|w|)` and each value is
  `clamp(round(w / scale), -1, 1)`. Values are stored as `int8` levels
  `{-1, 0, +1}` with FP32 scales in memory.
- QAT uses a straight-through estimator: the forward pass uses dequantized
  weights, the backward pass passes gradients to the full-precision master
  weights.
- `pack_ternary` packs four levels into one `uint8` using two bits per level:
  `00 -> -1`, `01 -> 0`, `10 -> +1`, `11 -> padding`. `unpack_ternary` restores
  levels and drops padding.
- Theoretical cost with FP16 scales and group size 128 is
  `2 + 16 / 128 = 2.125 bit/weight` (~21.25 GB for 80B parameters), excluding
  alignment, metadata and non-quantized embeddings/routers.

### On-disk AURORA-TQ layout (`*.aurora-tq`, version 1)

`aurora.quantization.tq` implements a versioned ZIP container (ZIP magic bytes)
with `ZIP_STORED` entries:

| Entry | Meaning |
| --- | --- |
| `metadata.json` | Format/version, full experiment config, quantization settings, tensor table and parameter counts |
| `tensors/NNNNN.levels` | Packed ternary levels (2 bits per value, `uint8`) |
| `tensors/NNNNN.scales` | Group scales (`float16` by default, `float32` optional) |
| `tensors/NNNNN.data` | Dense tensor bytes (embeddings, norms, biases, routers and skipped layers) |

Tensor metadata records canonical name, source state-dict name, kind, shape,
dtype or group size, entry names and SHA-256 checksums. Loading validates the ZIP
container, format version, metadata size, entry paths, checksums, shapes, packed
level size, scales size, duplicate names and unknown kinds. Writes are atomic.

`metadata.quantization` also records `calibrated` and `calibration_steps`. With
`--calibrate`, group scales are refined offline by alternating ternary
assignments and the least-squares scale `sum(x*q) / sum(q*q)`; the scale with the
lowest per-group reconstruction error is kept, so it is never worse than the
mean-absolute baseline. Calibration only changes scale values, not the archive
layout or format version. It uses weights, not activations.

```bash
python -m aurora.quantization.tq export \
  --checkpoint runs/aurora_100m_qat/best.pt \
  --output model.aurora-tq
python -m aurora.quantization.tq inspect model.aurora-tq
```

`load_tq_model` dequantizes ternary tensors into a regular dense model. This path
is for validation and compatibility, not for the future memory-mapped runtime. A
fast ternary matmul and a memory-mapped expert loader are still required before
this format becomes the production runtime path.

## Evaluation report (`eval.json`)

`aurora.benchmarks.evaluate` writes:

| Key | Meaning |
| --- | --- |
| `created_utc` | Report timestamp |
| `run_metadata` | Full checkpoint run metadata: run id, git, environment and dataset provenance |
| `checkpoint.path` | Checkpoint path |
| `checkpoint.step` | Optimizer step stored in the checkpoint |
| `checkpoint.qat_active` | Whether QAT conversion was active |
| `checkpoint.architecture` | `aurora` or `transformer` |
| `checkpoint.total_parameters` | Total parameter count of the loaded model |
| `checkpoint.run_id` | Run id from the checkpoint |
| `checkpoint.dataset_versions` | Training dataset versions recorded in the checkpoint |
| `checkpoint.metrics` | Last train/validation metrics stored in the checkpoint |
| `evaluation.git` | Git state of the evaluation process |
| `evaluation.environment` | Python, platform, Torch, CUDA and device info |
| `dataset_versions` | Sorted unique manifest versions of the evaluated files |
| `settings` | Sequence length, batch size, batch limit, device, AMP mode and verification flag |
| `results.<domain>` | `path`, `tokens`, `batches`, `loss`, `perplexity`, `dataset` and optional `dataset_version`/`manifest` |

If a `manifest.json` from `aurora.data.prepare` is found next to the token file,
the report verifies its `dataset_version`, split size and checksum, then includes
`dataset_version`, the split name, token count and SHA-256 of the evaluated file.
Use `--no-verify-data` to skip rehashing while still keeping the manifest
metadata.

## Zero-shot report (`zero_shot.json`)

`aurora.benchmarks.zero_shot` reads one JSONL file per task. Each record needs a
`context` string, a non-empty `choices` list of non-empty strings and an integer
`answer` index. Field names are configurable with `--context-field`,
`--choices-field` and `--answer-field`.

| Key | Meaning |
| --- | --- |
| `created_utc` | Report timestamp |
| `run_metadata` | Full checkpoint run metadata |
| `checkpoint` | Path, step, QAT flag, architecture, parameter count, run id and stored metrics |
| `evaluation` | Git state and environment |
| `settings` | Task paths, tokenizer, field names, maximum length, limit, normalization, device and AMP mode |
| `tasks.<name>` | Sample count, evaluated/skipped counts, total choices, correct count and accuracy |

The score of a choice is the sum of log-probabilities of its tokens conditioned
on the context; `--normalize-by-length` divides by the number of scored tokens.
Context is truncated from the left when `context + choice` exceeds
`--max-length`, then the choice is truncated if needed. Samples whose choices
cannot be scored after truncation are counted in `skipped` and excluded from
accuracy. `--predictions` writes one JSONL row per evaluated sample with `task`,
`index`, `answer`, `prediction`, `correct` and per-choice `scores`.

The harness expects preprocessed multiple-choice files; it does not download
MMLU/HellaSwag/ARC/WinoGrande or assemble few-shot prompts. Choices are scored in
separate forward passes, so evaluation is sequential rather than batched.

## Results summary (`results.md`, `results.json`)

`aurora-report` combines any number of eval and zero-shot reports:

```bash
python -m aurora.report \
  --report seed0=reports/eval_seed0.json \
  --report seed1=reports/eval_seed1.json \
  --output reports/results.md --json reports/results.json
```

| Key | Meaning |
| --- | --- |
| `format_version` | Summary schema version (currently `1`) |
| `created_utc` | Creation timestamp |
| `reports` | Per-input name, path, detected kind, values and run metadata |
| `reports[].run_id` | Run id from `run_metadata` |
| `reports[].git_commit`, `git_branch`, `git_dirty` | Git state, preferring the checkpoint run metadata and falling back to the report's evaluation block |
| `reports[].checkpoint_step` | Step stored in the evaluated checkpoint |
| `reports[].dataset_versions` | Dataset versions from the report or checkpoint |
| `metrics` | One entry per task/metric with `count`, `mean`, `std`, `min`, `max` and raw `values` |
| `metrics[].higher_is_better` | `false` for perplexity, `true` for accuracy |
| `limitations` | Statistical and comparability caveats |

Input kind is detected from the report shape: `results.<domain>.perplexity` is
an eval report and `tasks.<task>.accuracy` is a zero-shot report. Each report is
treated as one independent run. `std` is the sample standard deviation
(`ddof=1`) and is `0` for a single report. The Markdown output contains the
summary table, one row per input run with run id/git/step/dataset versions, and
the limitations.

## Quantization comparison report (`quantization.json`)

`aurora.benchmarks.quantization` writes one report for all weight-only variants:

| Key | Meaning |
| --- | --- |
| `created_utc` | Report timestamp |
| `run_metadata` | Full checkpoint run metadata |
| `checkpoint` | Path, step, QAT flag, architecture, parameter count and run id |
| `evaluation` | Git state and environment of the comparison process |
| `dataset` | Resolved evaluation file: SHA-256, split, tokens, manifest and `dataset_version` |
| `settings` | Sequence length, batch size, batch limit, group size, calibration steps, `min_features`, skip prefixes and device |
| `results.<variant>` | Loss, perplexity, token/batch counts and optional `quantization` stats |
| `results.<variant>.quantization` | Modules, parameters, weight MSE, dense bytes, estimated bytes and `bits_per_weight` (absent or `null` for FP32) |

Variants are weight-only fake quantization (`fp32`, `fp16`, `ternary`,
`ternary_calibrated`, `int8`, `int4`). They replace eligible `nn.Linear` weights
before evaluation, so the report measures quality and storage estimates, not
runtime kernel speed. `fp32` is always included as the baseline; other variants
also receive `perplexity_delta` and `perplexity_ratio` relative to it. A variant
is reported as skipped when no layer meets `min_features`/skip rules. QAT
checkpoints are rejected because their `TernaryLinear` path would re-quantize
weights during the forward pass.

## State stability report (`state_stability.json`)

`aurora.benchmarks.state_stability` writes:

| Key | Meaning |
| --- | --- |
| `created_utc` | Report timestamp |
| `config`, `checkpoint` | Config path and optional checkpoint path |
| `architecture`, `device` | Model architecture and execution device |
| `evaluation` | Git state and environment |
| `settings` | Lengths, chunk size, reset interval, seed and optional data path |
| `peak_rss_mb` | Peak process RSS |
| `results` | One row per length |

Each result contains `length`, processed `chunks`, `scored_tokens`, `seconds`,
`tokens_per_second`, `finite`, `loss`, `perplexity`, `first_chunk_loss`,
`last_chunk_loss`, `state_norm_first`, `state_norm_final`, `state_norm_max`,
`state_max_abs`, `state_growth` and the effective `reset_interval`. State tensors
are taken generically from either the AURORA block state or the Transformer KV
cache.

## Component ablation report (`ablation.json`)

`aurora.benchmarks.ablation` writes:

| Key | Meaning |
| --- | --- |
| `created_utc` | Report timestamp |
| `architecture` | Model architecture (must be `aurora`) |
| `evaluation` | Git state and environment |
| `dataset` | Train/validation sources with SHA-256 and `dataset_version` |
| `settings` | Variants, steps, sequence/batch sizes, token budget, device, seed and paths |
| `results.<variant>` | Flags, run id, output paths, parameter counts, training and validation metrics |

Variants are `core` (`use_memory=false`, `use_moe=false`), `core_memory`
(`use_memory=true`, `use_moe=false`) and `full` (both enabled). Every variant is
trained from scratch with the same seed, data and token budget. When `core` is
present, the other variants also receive `loss_delta_vs_core` and
`perplexity_ratio_vs_core`. Disabled branches remove their parameters, state and
losses rather than masking them at runtime.
