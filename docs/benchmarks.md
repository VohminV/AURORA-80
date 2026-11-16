# Baseline benchmarks

These measurements are reference points for the current prototype, not claims of
architectural quality. The smoke model is deliberately tiny and undertrained, and
the 100M measurements below were taken on CPU and on an 8 GB CUDA GPU.

## Environment

CPU measurements:

- CPU: 8 vCPU (x86_64), 31 GiB RAM
- PyTorch: 2.15.0+cpu
- Python: 3.12.3
- CUDA: not used

CUDA measurements:

- GPU: NVIDIA GeForce RTX 3050, 7,885 MiB, compute capability 8.6
- Driver: 615.71.09; PyTorch 2.14.1+cu130 (CUDA 13.0)
- Python: 3.12.3
- Separate `.venv-cuda`; the main `.venv` stays CPU-only
- `inductor` needs `Python.h`, which is not installed system-wide. The CUDA
  runs used locally extracted Python 3.12 headers via `CPATH`; this is a host
  workaround, not a project dependency

Reproduce CPU numbers with:

```bash
python scripts/benchmark.py \
  --config configs/aurora_100m.yaml \
  --seq-len 512 \
  --batch-size 1 \
  --train-steps 1 \
  --inference-steps 2 \
  --generation-tokens 2 \
  --output reports/benchmark_100m_cpu.json
```

## AURORA-100M, CPU reference

| Metric | Value |
| --- | ---: |
| Total parameters | 105,727,560 |
| Active parameters | 49,104,456 |
| Sequence length | 512 |
| Batch size | 1 |
| Train throughput | 46.3 tokens/s |
| Train step time | 11.06 s |
| Inference throughput | 213.2 tokens/s |
| Inference latency | 4.69 ms/token |
| Generation throughput | 7.2 tokens/s |
| Generation latency | 138.1 ms/token |
| Peak RSS (training process) | 4,519 MB |
| Peak VRAM | not measured (no CUDA) |

The generation number uses state continuation without a KV-cache. Each generated
token requires a full forward through the recurrent core and memory, so generation
is an order of magnitude slower than batched prefill. Optimized kernels, chunkwise
parallel scan and a custom runtime are not implemented yet.

## Transformer-100M baseline, CPU reference

Configuration: `configs/transformer_100m.yaml` (106.28M total parameters, RoPE,
SwiGLU, KV-cache). The same command is used with the baseline config:

```bash
python scripts/benchmark.py \
  --config configs/transformer_100m.yaml \
  --seq-len 512 \
  --batch-size 1 \
  --train-steps 1 \
  --inference-steps 2 \
  --generation-tokens 2 \
  --output reports/benchmark_transformer_100m_cpu.json
```

| Metric | AURORA-100M | Transformer-100M |
| --- | ---: | ---: |
| Total parameters | 105,727,560 | 106,284,288 |
| Active parameters | 49,104,456 | 106,284,288 |
| Train throughput | 46.3 tokens/s | 272.9 tokens/s |
| Train step time | 11.06 s | 1.88 s |
| Inference throughput | 213.2 tokens/s | 1,475.7 tokens/s |
| Inference latency | 4.69 ms/token | 0.68 ms/token |
| Generation throughput | 7.2 tokens/s | 28.2 tokens/s |
| Peak RSS (training process) | 4,519 MB | 2,268 MB |

The gap is a property of the current implementation, not evidence about model
quality: AURORA's recurrent core and associative memory iterate over tokens in a
Python loop, while the Transformer uses fused scaled-dot-product attention.
Closing this gap requires the chunkwise parallel scan and fused kernels listed
in the roadmap. These numbers are CPU-only and must not be presented as GPU
throughput.

## AURORA-100M, CUDA, AMP

The CUDA environment uses `torch.autocast` with BF16 (`auto` resolves to BF16 on
this GPU). The FP16 row additionally exercises `GradScaler`.

```bash
.venv-cuda/bin/python scripts/benchmark.py \
  --config configs/aurora_100m.yaml --device cuda \
  --seq-len 512 --batch-size 1 \
  --train-steps 3 --inference-steps 3 --generation-tokens 4 \
  --output reports/benchmark_100m_cuda_seq512.json

sed 's/dtype_amp: auto/dtype_amp: fp16/' configs/aurora_100m.yaml \
  > /tmp/aurora_100m_fp16.yaml
.venv-cuda/bin/python scripts/benchmark.py \
  --config /tmp/aurora_100m_fp16.yaml --device cuda \
  --seq-len 64 --batch-size 1 \
  --train-steps 3 --inference-steps 3 --generation-tokens 4 \
  --output reports/benchmark_100m_cuda_fp16.json
```

| Metric | BF16, seq 512 | FP16 + GradScaler, seq 64 |
| --- | ---: | ---: |
| Total parameters | 105,727,560 | 105,727,560 |
| Active parameters | 49,104,456 | 49,104,456 |
| Train throughput | 22.0 tokens/s | 20.6 tokens/s |
| Train step time | 23.27 s | 3.11 s |
| Inference throughput | 81.1 tokens/s | 76.9 tokens/s |
| Inference latency | 12.33 ms/token | 13.00 ms/token |
| Generation throughput | 7.2 tokens/s | 7.3 tokens/s |
| Generation latency | 138.9 ms/token | 136.8 ms/token |
| Peak VRAM allocated | 2,671 MB | 2,033 MB |
| Peak VRAM reserved | 3,194 MB | 2,246 MB |
| Peak RSS | 3,180 MB | 2,505 MB |

CUDA is slower than the CPU reference for training and generation. The recurrent
core launches many small kernels per token, so launch overhead dominates; the
CPU measurement had more threads working on the same sequential loop. This is a
property of the reference implementation, not of the architecture. VRAM is not a
constraint at 100M: the model uses about 2.7 GB of the 7.9 GB available.

Gradient checkpointing on CUDA (same seq 512 command plus
`--gradient-checkpointing`) reduced peak allocated VRAM from 2,671 MB to
2,033 MB (24%) and reserved VRAM from 3,194 MB to 2,180 MB, at the cost of
throughput falling from 22.0 to 12.9 tokens/s (41%).

A short CUDA smoke run (`configs/smoke.yaml`, 10 steps, then resume to 15 steps)
trained at roughly 330–420 tokens/s and reduced validation loss from 5.3311 to
5.2728. AMP, checkpointing and resume work on CUDA.

## `torch.compile` on CUDA

```bash
.venv-cuda/bin/python scripts/benchmark.py \
  --config configs/aurora_100m.yaml --device cuda \
  --seq-len 64 --batch-size 1 \
  --train-steps 3 --inference-steps 3 --generation-tokens 4 \
  --compile --compile-backend aot_eager \
  --output reports/benchmark_100m_cuda_aot_eager.json
```

`inductor` uses the same command with `--compile-backend inductor` and the
local `CPATH` workaround described above.

| Backend | Train tokens/s | Inference tokens/s | Generation tokens/s |
| --- | ---: | ---: | ---: |
| eager | 20.6 | 75.5 | 11.2 |
| `aot_eager` | 1.6 | 50.6 | 0.1 |
| `inductor` | 0.4 | 300.7 | 0.04 |

`inductor` makes batched inference about 4× faster than eager at seq 64, but
training and generation are much slower. The MoE router creates dynamic expert
group shapes, so Dynamo hits the default recompile limit (8) in every compiled
run. Removing those recompilations is still required before compiled training or
generation can be used; CUDA graphs have not been tested.

## Component profile, AURORA-100M, CPU

```bash
python scripts/profile_components.py \
  --config configs/aurora_100m.yaml \
  --seq-len 32 --batch-size 1 --steps 2 --warmup-steps 1 \
  --threads 4 --output reports/profile_components_100m_cpu.json
```

| Component | Forward | Forward share | Backward | Backward share |
| --- | ---: | ---: | ---: | ---: |
| Recurrent core | 130.0 ms | 59.6% | 256.6 ms | 57.2% |
| Associative memory | 49.5 ms | 22.7% | not isolated | - |
| MoE | 23.2 ms | 10.6% | 51.6 ms | 11.5% |
| Whole step | 218.3 ms | 100% | 448.5 ms | 100% |

The remaining forward time is normalization, residual integration and dropout.
The remaining backward time includes the memory branch, which is invoked through
a direct method call and therefore is only timed in the forward pass. Hooks add
some overhead, so this is an approximate CPU breakdown, not a GPU profile.

## Smoke model baseline

Configuration: `configs/smoke.yaml` (289.48K total / 191.18K active parameters).
Corpus: `scripts/make_demo_corpus.py`, byte tokenizer, 3,369 train windows and
177 validation windows of length 32.

```bash
python scripts/make_demo_corpus.py --output data/demo.txt --lines 2000
python -m aurora.data.prepare \
  --input data/demo.txt --output-dir data/smoke \
  --tokenizer byte --val-fraction 0.05
python -m aurora.training.pretrain --config configs/smoke.yaml \
  --output-dir runs/smoke_baseline
python -m aurora.benchmarks.perplexity \
  --checkpoint runs/smoke_baseline/best.pt \
  --data data/smoke/val.bin --batch-size 4 --device cpu
```

| Metric | Value |
| --- | ---: |
| Optimizer steps | 50 |
| Train LM loss (last log) | 4.848 |
| Validation loss (step 25) | 4.953 |
| Validation loss (step 50) | 4.769 |
| Validation perplexity (step 50) | 117.76 |
| Validation perplexity, full file | 125.28 |
| Training throughput | ~1,400 tokens/s |

Generation from this checkpoint is not meaningful: after 50 steps on a synthetic
corpus the model emits byte garbage. The smoke run only verifies that the pipeline
learns and that checkpoints, resume, evaluation and inference work.

## Weight-only quantization smoke comparison (CPU)

This validates the comparison harness on the smoke checkpoint and synthetic
validation data. It is not evidence about final quality; the model is tiny and
undertrained.

```bash
python -m aurora.benchmarks.quantization \
  --checkpoint runs/smoke/best.pt \
  --data data/smoke/val.bin \
  --group-size 16 --min-features 16 \
  --batch-size 4 --batches 2 --seq-len 32 \
  --output reports/smoke_quantization.json
```

| Variant | Perplexity | Weight MSE | bit/weight |
| --- | ---: | ---: | ---: |
| FP32 | 111.41 | - | 32 |
| FP16 weights | 111.41 | 1.37e-11 | 16 |
| Ternary | 122.90 | 7.84e-05 | 3.00 |
| Ternary + calibration | 115.51 | 5.33e-05 | 3.00 |
| INT8 (group 16) | 111.42 | 7.06e-09 | 9.00 |
| INT4 (group 16) | 111.66 | 2.32e-06 | 5.00 |

Calibration recovers part of the ternary gap on this checkpoint. The numbers are
CPU fake-quantization results, not a runtime speed benchmark.

## State stability, smoke model, CPU

The stress test feeds deterministic random tokens through one continued state.
Random targets put the expected loss near `ln(260) = 5.56`, so this checks
numerical stability, not language quality.

```bash
python -m aurora.benchmarks.state_stability \
  --checkpoint runs/smoke/best.pt \
  --lengths 4096 8192 16384 32768 \
  --chunk-size 256 \
  --output reports/state_stability.json
```

| Length | Loss | max abs state | Final state norm | tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 5.5843 | 4.67e-01 | 2.395 | 1,525 |
| 8,192 | 5.5887 | 4.67e-01 | 2.297 | 1,481 |
| 16,384 | 5.5888 | 4.67e-01 | 2.376 | 1,499 |
| 32,768 | 5.5891 | 4.88e-01 | 2.329 | 1,529 |

All runs stayed finite with no state-norm growth. Resetting every 4,096 tokens
produced the same 32k loss and final norm within printed precision. The smoke
model is undertrained and configured for length 32, so these results only show
that the recurrence and memory do not explode numerically; 100M long-context
quality still requires training and evaluation.

## Component ablation, smoke model, CPU

The ablation trains `core`, `core_memory` and `full` from scratch with the same
seed, data and token budget. It answers a narrow question: do the memory and MoE
branches add parameters and state without breaking the forward pass? It is not
evidence that either component improves quality.

```bash
python -m aurora.benchmarks.ablation \
  --config configs/smoke.yaml \
  --variants core core_memory full \
  --output reports/smoke_ablation.json
```

| Variant | Total params | Active params | Val loss | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| core | 58,436 | 58,436 | 4.8282 | 124.99 |
| core_memory | 92,108 | 92,108 | 4.7667 | 117.53 |
| full | 289,484 | 191,180 | 4.7666 | 117.52 |

Disabling a branch removes its parameters and state: the core-only model has no
memory tensors and zero memory/consistency/aux losses. On 50 smoke steps the
memory branch gives a small loss improvement, while MoE adds 197k total
parameters for a negligible change. With random tokens and this budget the result
is noise; a meaningful ablation requires a trained 100M checkpoint and multiple
seeds.

## What is still missing

- Removing MoE recompilations under `torch.compile`. Both `aot_eager` and
  `inductor` hit the recompile limit on dynamic expert group sizes, so compiled
  training and generation are slower than eager.
- CUDA graphs. Only eager and `torch.compile` were measured.
- Multi-seed runs and confidence intervals.
- A trained 64k tokenizer and a real RU/EN/code corpus.
- A trained Transformer baseline under the same data, tokens and compute; only
  the implementation and parameter parity exist so far.
- Mamba/RWKV baselines.
- Ablations for `memory_loss_coef` and `consistency_loss_coef` on a trained
  checkpoint.
- Public benchmark scores and an evaluation report from real checkpoints.
