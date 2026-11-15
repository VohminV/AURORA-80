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
| `tokenizer` | Kind, model path and vocabulary size |
| `encoding`, `dtype`, `val_fraction`, `chunk_size_bytes` | Preparation settings |
| `inputs` | Per-file path, byte size, SHA-256 and token count |
| `outputs` | Per-split file name, byte size, SHA-256 and token count |

The `.bin` files are raw little-endian integer arrays (`uint16` or `uint32`),
without a header. Their shape is `(num_tokens,)` and the dtype is recorded in the
manifest and the training config.

## AURORA-TQ (ternary weights)

Implemented in `aurora/quantization/ternary.py` and `qat.py`.

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

### Proposed on-disk AURORA-TQ layout (not implemented)

The current code packs levels and keeps scales in memory but does not define a
stable file format. A future `*.aurora-tq` file should contain:

1. Magic bytes and format version.
2. JSON metadata: model config, quantization group size, dtype, tensor names,
   shapes, strides and checksums.
3. Per-tensor packed level stream followed by group scales.
4. Optional per-tensor min/max calibration metadata.

A fast ternary matmul and a memory-mapped expert loader are required before this
format becomes the runtime path. Until then, QAT checkpoints remain regular
PyTorch checkpoints and inference uses the PyTorch dequantization path.

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
