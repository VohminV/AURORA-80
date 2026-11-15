# AURORA-80

**Adaptive Unified Recurrent and Associative Neural Architecture**

Исследовательский проект собственной языковой модели: без Transformer и self-attention, с многоуровневой рекуррентной динамикой, ассоциативной памятью и адаптивной маршрутизацией экспертов.

В репозитории реализован рабочий прототип на PyTorch: архитектура, обучение, токенизация, подготовка данных, ternary-квантование (QAT), генерация, оценка perplexity и тесты.

> **Статус:** AURORA-100M — исследовательский прототип. Архитектура ещё не доказала превосходство над Transformer. Целевая модель AURORA-80B требует отдельной вычислительной инфраструктуры.

---

## Ключевые характеристики

| Параметр | Значение |
| --- | --- |
| Общая ёмкость цели | 80 млрд параметров |
| Активные параметры цели | ~11.5 млрд (конфигурация 80B) |
| Архитектура | Non-Transformer, без self-attention |
| Целевая VRAM инференса | 8 GB (RTX 3050) |
| Целевая RAM инференса | 32 GB |
| Языки | Русский, английский |
| Квантование | AURORA-TQ, ternary + group scales |
| Runtime | PyTorch-прототип; C++/CUDA runtime — следующий этап |

---

## Реализованная архитектура

Каждый блок AURORA состоит из трёх ветвей с residual-интеграцией и LayerScale:

```
Tokenizer / Embedding
        ↓
┌─────────────────────────────────────┐
│ Recurrent Cognitive Core            │
│  - S временных масштабов            │
│  - gated state update               │
│  - межуровневое взаимодействие      │
├─────────────────────────────────────┤
│ Associative Memory                  │
│  - multi-head delta rule            │
│  - decay и write gates              │
│  - чтение по содержимому            │
├─────────────────────────────────────┤
│ Adaptive Experts (MoE)              │
│  - Top-K routing                    │
│  - load balancing + router z-loss   │
├─────────────────────────────────────┤
│ Residual Integration                │
└─────────────────────────────────────┘
        ↓
Language Prediction Head
```

### Рекуррентное ядро

Для каждого уровня `i`:

```
z_t^(i) = tanh(W_i x_t + U_i h_{t-1}^(i))
g_t^(i) = sigmoid(G_i x_t + R_i h_{t-1}^(i))
h_t^(i) = g_t^(i) * h_{t-1}^(i) + (1 - g_t^(i)) * z_t^(i)
```

После обновления уровней состояние смешивается обучаемыми проекциями и RMSNorm. Состояние хранится в FP32 для численной устойчивости даже при mixed precision.

### Ассоциативная память

На каждую голову поддерживается матрица памяти `M ∈ R^(d_head × d_head)`:

```
r_t = q_t^T M_{t-1}
M_t = λ_t M_{t-1} + η_t * k_t (v_t - M_{t-1}^T k_t)^T / (ε + ||k_t||²)
```

`λ_t` и `η_t` вычисляются из текущего токена. Ключи и запросы L2-нормируются. Память обучается end-to-end через языковую функцию потерь.

### Эксперты

- `N` экспертов SwiGLU на блок.
- Маршрутизатор Top-K с нормализованными весами.
- Switch-style load-balancing loss и router z-loss.
- Dispatch через сортировку назначений и групповые срезы.

В конфигурации 80B используется 20 экспертов на слой × 7 слоёв = 140 экспертов; на каждом слое активируются 2 из 20.

### Функция потерь

```
L = L_LM + α * L_balance + β * L_z + QAT-регуляризация через STE
```

`L_LM` — cross-entropy на следующий токен. Балансировка и z-loss уже включены в конфигурацию.

---

## Конфигурации

| Конфиг | Всего параметров | Активных | Назначение |
| --- | ---: | ---: | --- |
| `configs/smoke.yaml` | 0.29M | 0.19M | Проверка пайплайна |
| `configs/aurora_100m.yaml` | 105.73M | 49.10M | Первый прототип |
| `configs/aurora_1b.yaml` | 929.49M | 325.51M | Масштабирование |
| `configs/aurora_80b.yaml` | 80.49B | 11.52B | Целевая конфигурация |

Посчитать параметры без выделения памяти:

```bash
python scripts/count_parameters.py --config configs/aurora_80b.yaml
```

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tokenizer,dev]"
```

Для CUDA рекомендуется установить PyTorch под свою версию CUDA заранее:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Проверка:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Быстрый старт

### 1. Демонстрационный корпус и smoke-обучение

```bash
python scripts/make_demo_corpus.py --output data/demo.txt --lines 2000
python -m aurora.data.prepare \
  --input data/demo.txt \
  --output-dir data/smoke \
  --tokenizer byte \
  --val-fraction 0.05
python -m aurora.training.pretrain --config configs/smoke.yaml
```

После обучения чекпоинты появятся в `runs/smoke/`:

- `latest.pt` — последний чекпоинт;
- `best.pt` — лучший по validation loss.

### 2. Генерация

```bash
python -m aurora.generation \
  --checkpoint runs/smoke/best.pt \
  --prompt "AURORA" \
  --tokenizer byte \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.95 \
  --device cpu
```

### 3. Perplexity

```bash
python -m aurora.benchmarks.perplexity \
  --checkpoint runs/smoke/best.pt \
  --data data/smoke/val.bin \
  --batch-size 4 \
  --batches 20 \
  --device cpu
```

---

## Обучение на реальном корпусе

### Токенизатор

```bash
python -m aurora.tokenizer.train_tokenizer \
  --input corpus_en.txt corpus_ru.txt corpus_code.txt \
  --model-prefix tokenizer/aurora_64k \
  --vocab-size 64000 \
  --model-type bpe
```

### Подготовка данных

```bash
python -m aurora.data.prepare \
  --input corpus_en.txt corpus_ru.txt corpus_code.txt \
  --output-dir data/processed \
  --tokenizer sentencepiece \
  --tokenizer-model tokenizer/aurora_64k.model \
  --dtype uint16
```

### Запуск AURORA-100M

```bash
python -m aurora.training.pretrain --config configs/aurora_100m.yaml
```

Переопределение путей и параметров:

```bash
python -m aurora.training.pretrain \
  --config configs/aurora_100m.yaml \
  --train-path data/processed/train.bin \
  --val-path data/processed/val.bin \
  --output-dir runs/aurora_100m_run1 \
  --max-steps 20000 \
  --device cuda
```

### Возобновление обучения

```bash
python -m aurora.training.pretrain \
  --config configs/aurora_100m.yaml \
  --resume runs/aurora_100m/latest.pt
```

### QAT с ternary-весами

```bash
python -m aurora.training.pretrain --config configs/aurora_100m_qat.yaml
```

По умолчанию QAT включается на шаге, указанном в `qat_start_step`. Формат AURORA-TQ использует веса `{-s, 0, +s}` и групповые FP16-масштабы. При группе 128 теоретическая стоимость — около `2.125 bit/weight` (примерно 21.25 GB для 80B без накладных расходов).

---

## Что реализовано

- Многоуровневое рекуррентное ядро с межуровневым взаимодействием.
- Ассоциативная память с delta rule, decay и write gates.
- MoE с Top-K, load balancing и z-loss.
- Residual integration со LayerScale.
- Инициализация, RMSNorm, tied embeddings, SwiGLU-эксперты.
- AdamW с раздельным weight decay.
- Cosine scheduler с warmup.
- BF16/FP16 autocast, GradScaler для FP16.
- Gradient accumulation, clipping и пропуск нефинитных шагов.
- Атомарные чекпоинты, resume, best/latest.
- Validation loss и perplexity.
- SentencePiece и byte-level tokenizer.
- Memmap-датасет для токенизированных корпусов.
- Ternary-квантование, упаковка 2-bit и QAT через straight-through estimator.
- Загрузка чекпоинтов для инференса с автоматическим включением ternary-режима.
- Генерация с temperature, top-k, top-p и repetition penalty.
- Автоматический BF16/FP16 autocast на CUDA при генерации и оценке perplexity.
- 38 тестов: причинность, состояния, память, маршрутизация, квантование, обучение, resume, QAT и инференс.

---

## Структура репозитория

```
AURORA-80/
├── aurora/
│   ├── architecture/
│   │   ├── recurrent_core.py
│   │   ├── associative_memory.py
│   │   ├── expert_router.py
│   │   ├── integration.py
│   │   ├── language_head.py
│   │   ├── layers.py
│   │   ├── block.py
│   │   ├── state.py
│   │   └── model.py
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   ├── tokenizer/
│   │   ├── tokenizer.py
│   │   └── train_tokenizer.py
│   ├── data/
│   │   ├── prepare.py
│   │   └── dataset.py
│   ├── quantization/
│   │   ├── ternary.py
│   │   └── qat.py
│   ├── benchmarks/
│   │   └── perplexity.py
│   ├── config.py
│   ├── generation.py
│   ├── inference.py
│   └── utils.py
├── configs/
├── scripts/
├── tests/
├── pyproject.toml
└── README.md
```

---

## Проверка качества кода

```bash
pytest
ruff check aurora tests scripts
ruff format --check aurora tests scripts
mypy aurora
```

Текущий статус: все тесты проходят, `ruff` и `mypy` не находят ошибок.

---

## Ограничения прототипа

- Рекуррентное ядро и память обрабатывают токены последовательно. Это упрощает верификацию, но ограничивает скорость обучения. Следующий шаг — chunkwise parallel scan и/или `torch.compile`.
- Вспомогательные `L_memory` и `L_consistency` из исходной концепции пока не реализованы: память обучается через `L_LM`.
- 80B-конфигурация не обучалась и не запускалась. Это расчётная конфигурация.
- Экспертная система распределена по слоям; стратегия разделяемых экспертов для 80B требует отдельных экспериментов.
- `prepare.py` читает корпус и хранит токены в памяти. Для корпусов масштаба 80B нужен потоковый пайплайн с записью в memmap.
- Нет распределённого обучения, C++/CUDA runtime и собственного формата сериализации AURORA-TQ.
- Пропускная способность RAM и предсказательная загрузка экспертов для 8 GB VRAM ещё не реализованы.

---

## Научная база

- [Mamba-2](https://arxiv.org/abs/2405.21060) — эффективное рекуррентное моделирование.
- [RWKV-7](https://arxiv.org/abs/2503.14456) — динамическое обновление состояния.
- [Titans](https://arxiv.org/abs/2501.00663) — обучаемая долговременная память.
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — сверхнизкая точность весов.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — масштабирование экспертных подсетей.
- [Chinchilla](https://arxiv.org/abs/2203.15556) — compute-optimal обучение.

Исследовательская гипотеза: сочетание многоуровневой рекуррентной динамики, ассоциативной памяти и адаптивной маршрутизации экспертов может дать качественное моделирование языка без Transformer Attention. Превосходство над существующими архитектурами пока не доказано.

---

## Лицензия

Не определена. Проект находится на стадии исследования.
