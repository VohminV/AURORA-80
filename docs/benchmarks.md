# Baseline benchmarks

These measurements are reference points for the current prototype, not claims of
architectural quality. The smoke model is deliberately tiny and undertrained, and
the 100M measurements below were taken on CPU only.

## Environment

- CPU: 8 vCPU (x86_64), 31 GiB RAM
- PyTorch: 2.15.0+cpu
- Python: 3.12.3
- CUDA: not available in this environment
- Commit: the commit that introduces this file

Reproduce with:

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
Python loop, while the Transformer uses fused scaled dot-product attention.
Closing this gap requires the chunkwise parallel scan and fused kernels listed
in the roadmap. These numbers are CPU-only and must not be presented as GPU
throughput.

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

## What is still missing

- CUDA AMP and production `torch.compile` measurements. `aot_eager` completes on
  CPU, but `inductor` is blocked because this environment lacks `Python.h`.
- Reducing MoE recompilations under `torch.compile` (dynamic expert group sizes
  hit the recompile limit).
- Multi-seed runs and confidence intervals.
- A trained 64k tokenizer and a real RU/EN/code corpus.
- A trained Transformer baseline under the same data, tokens and compute; only
  the implementation and parameter parity exist so far.
- Mamba/RWKV baselines.
- Ablations for `memory_loss_coef` and `consistency_loss_coef`.
- Public benchmark scores and an evaluation report from real checkpoints.
