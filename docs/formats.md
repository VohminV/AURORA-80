# Checkpoint and AURORA-TQ formats

## Checkpoint format (`*.pt`)

Checkpoints are written with `torch.save` and loaded with
`torch.load(..., weights_only=True)`. Loading therefore never unpickles arbitrary
Python objects.

Top-level keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `format_version` | `int` | Checkpoint schema version (currently `1`) |
| `model` | `dict[str, Tensor]` | Model `state_dict` |
| `optimizer` | `dict` | AdamW state |
| `scheduler` | `dict` | Cosine scheduler state |
| `scaler` | `dict \| None` | FP16 `GradScaler` state, `None` outside FP16 CUDA |
| `step` | `int` | Number of completed optimizer steps |
| `config` | `dict` | Full experiment config (`model` and `training` sections) |
| `qat_active` | `bool` | Whether ternary QAT conversion is active |
| `best_val` | `float \| None` | Best validation loss seen so far |
| `rng_state` | `dict \| None` | Torch CPU/CUDA RNG state for resume |

Properties:

- Writes are atomic: data is saved to `<name>.tmp` and moved with `os.replace`.
- `model`, `optimizer`, `scheduler`, `scaler`, `step`, `config`, `qat_active` and
  `best_val` are required to resume. `rng_state` is optional.
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
| `checkpoint.path` | Checkpoint path |
| `checkpoint.step` | Optimizer step stored in the checkpoint |
| `checkpoint.qat_active` | Whether QAT conversion was active |
| `checkpoint.architecture` | `aurora` or `transformer` |
| `checkpoint.total_parameters` | Total parameter count of the loaded model |
| `settings` | Sequence length, batch size, batch limit, device and AMP mode |
| `results.<domain>` | `path`, `tokens`, `batches`, `loss`, `perplexity` and optional manifest metadata |

If a `manifest.json` from `aurora.data.prepare` is found next to the token file,
the report includes the split name, token count and SHA-256 of the evaluated
file, which ties the metric to a dataset version.
