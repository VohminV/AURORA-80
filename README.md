# AURORA-80

**Adaptive Unified Recurrent and Associative Neural Architecture**

Исследовательский проект собственной языковой модели: без Transformer и self-attention, с многоуровневой рекуррентной динамикой, ассоциативной памятью и адаптивной маршрутизацией экспертов.

В репозитории реализован рабочий прототип на PyTorch: архитектура, обучение, токенизация, потоковая подготовка данных, ternary-квантование (QAT), генерация, оценка perplexity, единый eval harness, Transformer baseline для честного сравнения и тесты.

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
L = L_LM + α * L_balance + β * L_z + γ * L_memory + δ * L_consistency
    + QAT-регуляризация через STE
```

`L_LM` — cross-entropy на следующий токен. `L_memory` — ошибка предсказания значения до записи в память, `L_consistency` — ошибка чтения сразу после записи. Оба aux-лосса включаются коэффициентами `memory_loss_coef` и `consistency_loss_coef`; в 100M-конфиге они равны `0.01`.

---

## Конфигурации

| Конфиг | Всего параметров | Активных | Назначение |
| --- | ---: | ---: | --- |
| `configs/smoke.yaml` | 0.29M | 0.19M | Проверка пайплайна |
| `configs/aurora_100m.yaml` | 105.73M | 49.10M | Первый прототип |
| `configs/transformer_100m.yaml` | 106.28M | 106.28M | Transformer baseline |
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

Батчевая и streaming-генерация доступны через API:

```python
from aurora.generation import generate_batch, stream_generate

completions, state = generate_batch(model, prompts, max_new_tokens=64, temperature=0.8)
for step in stream_generate(model, prompts, max_new_tokens=64, temperature=0.8):
    ...  # step[i] — токен i-й строки или None, если строка завершилась
```

### 3. HTTP API

```bash
python -m aurora.server --checkpoint runs/smoke/best.pt --port 8000
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "AURORA", "max_new_tokens": 32, "temperature": 0.8, "top_k": 20, "top_p": 0.95}'
```

Сервер по умолчанию слушает `127.0.0.1` и не имеет аутентификации: не открывайте его в публичную сеть. Endpoint `/generate` принимает один prompt и возвращает `text`, `tokens`, `prompt_tokens` и `finish_reason`; тело запроса ограничено 1 MiB, prompt — `--max-prompt-tokens` (по умолчанию 4096). Батчевый и streaming-режимы доступны через Python API.

### 4. Perplexity

```bash
python -m aurora.benchmarks.perplexity \
  --checkpoint runs/smoke/best.pt \
  --data data/smoke/val.bin \
  --batch-size 4 \
  --batches 20 \
  --device cpu
```

### 5. Единый eval harness

```bash
python -m aurora.benchmarks.evaluate \
  --checkpoint runs/smoke/best.pt \
  --data ru=data/processed/ru_val.bin \
  --data en=data/processed/en_val.bin \
  --data code=data/processed/code_val.bin \
  --output reports/eval.json
```

Отчёт содержит loss и perplexity по каждому домену, число токенов, шаг чекпоинта, архитектуру и SHA-256 из `manifest.json`, если он есть рядом с `.bin`.

---

## Обучение на реальном корпусе

### Курирование корпуса

```bash
python -m aurora.data.curate \
  --input corpus_ru.txt corpus_en.jsonl corpus_code.txt \
  --output data/curated/corpus.txt \
  --report data/curated/curation_report.json \
  --rejects data/curated/rejects.jsonl
```

`curate.py` читает текст и JSONL потоково, нормализует Unicode/переводы строк и пропускает только документы, прошедшие фильтры качества: длину в символах и словах, долю букв/цифр/символов, управляющие символы, длинные повторы и долю повторяющихся строк. Точные дубликаты определяются по нормализованному тексту, почти-дубликаты — по 64-битному SimHash с 8 бэндами (`--near-dedup-threshold`, по умолчанию 3). PII (email, IPv4, номера карт) можно удалять (`--pii-action drop`), заменять маркерами (`redact`) или оставлять (`keep`).

Для text-входа документы разделяются пустой строкой; для JSONL текст берётся из поля `--json-field` (по умолчанию `text`), а битые записи учитываются в отчёте. `--report` содержит счётчики по каждому фильтру, распределение письменностей, SHA-256 входов/выхода, детерминированный `dataset_version` и `run_metadata`. `--rejects` пишет JSONL с причиной и позицией отклонённого документа; в режиме `drop` текст в preview не попадает. Курированный `.txt` затем передаётся в обучение токенизатора и `prepare.py`.

Текущая реализация хранит множество точных хэшей и SimHash-индексы в RAM. Для корпусов на миллиарды токенов понадобится sharded/on-disk дедупликация.

### Токенизатор

```bash
python -m aurora.tokenizer.train_tokenizer \
  --input corpus_en.txt corpus_ru.txt corpus_code.txt \
  --model-prefix tokenizer/aurora_64k \
  --vocab-size 64000 \
  --model-type bpe
```

### Оценка токенизатора

```bash
python -m aurora.tokenizer.stats \
  --input corpus_ru.txt corpus_en.txt corpus_code.txt \
  --tokenizer sentencepiece \
  --tokenizer-model tokenizer/aurora_64k.model \
  --output reports/tokenizer_stats.json
```

Отчёт содержит bytes/token, chars/token, tokens/word и абсолютные счётчики по каждому файлу. Это позволяет сравнить fertility для RU/EN/code до запуска обучения.

### Подготовка данных

```bash
python -m aurora.data.prepare \
  --input corpus_en.txt corpus_ru.txt corpus_code.txt \
  --output-dir data/processed \
  --tokenizer sentencepiece \
  --tokenizer-model tokenizer/aurora_64k.model \
  --dtype uint16
```

`prepare.py` читает входные файлы чанками (`--chunk-size`, по умолчанию 1 MiB), декодирует UTF-8 инкрементально и пишет `manifest.json` с SHA-256 входов, выходов и токенизатора. Поле `dataset_version` — детерминированный content hash от содержимого, split-настроек и идентичности токенизатора; повторная подготовка тех же данных даёт ту же версию. Весь корпус в RAM не загружается.

`--split-mode document` назначает в train/val/test целые документы (разделитель — пустая строка) и не разрезает их между сплитами. `--test-fraction` создаёт дополнительный `test.bin`; сумма val/test долей должна быть меньше 1. В document-режиме нужны документы для каждого непустого сплита, иначе подготовка завершится ошибкой вместо нарушения границ.

### Data card

```bash
python -m aurora.data.card \
  --manifest data/processed/manifest.json \
  --output data/processed/DATA_CARD.md \
  --name "AURORA RU/EN/code" \
  --description "Tokenized pretraining corpus" \
  --language ru --language en --language code \
  --license "TBD" \
  --verify
```

Карточка содержит `dataset_version`, таблицы источников и сплитов с SHA-256, настройки подготовки и токенизатора. `--verify` дополнительно пересчитывает размеры и checksums `train.bin`/`val.bin`/`test.bin`; `read_manifest` отвергает manifest с изменёнными метаданными, потому что версия не совпадает с content hash.

### Смешивание доменов

Каждый домен готовится отдельным запуском `prepare.py`, после чего веса задаются в конфиге:

```yaml
training:
  train_paths:
    - data/processed/ru/train.bin
    - data/processed/en/train.bin
    - data/processed/code/train.bin
  train_weights: [0.6, 0.3, 0.1]
  val_path: data/processed/val.bin
```

Готовый пример: `configs/aurora_100m_mixture.yaml`. Сэмплер тянет примеры с replacement, поэтому ожидаемая доля домена за эпоху равна нормализованному весу. `train_path` и `train_paths` взаимоисключающие.

Для каждого `.bin` trainer и eval автоматически ищут `manifest.json` рядом с файлом, проверяют `dataset_version`, размер и SHA-256, а также dtype. Если manifest лежит отдельно, его можно передать явно:

```bash
python -m aurora.training.pretrain \
  --config configs/aurora_100m_mixture.yaml \
  --dataset-manifest data/processed/ru/manifest.json \
  --dataset-manifest data/processed/en/manifest.json \
  --dataset-manifest data/processed/code/manifest.json
```

Пути train/val/mixture должны быть описаны в переданных manifest-файлах. `--no-verify-data` пропускает повторное хеширование, но оставляет проверку версии manifest и записанных checksums.

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

`--gradient-checkpointing` включает activation checkpointing, `--max-tokens` останавливает обучение по бюджету обработанных токенов, а `--early-stopping-patience` — после заданного числа validation-замеров без улучшения (требуется `--val-path`).

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

Экспорт и инспекция версионированного архива:

```bash
python -m aurora.quantization.tq export \
  --checkpoint runs/aurora_100m_qat/best.pt \
  --output model.aurora-tq
python -m aurora.quantization.tq inspect model.aurora-tq
```

Экспорт с калибровкой групповых масштабов:

```bash
python -m aurora.quantization.tq export \
  --checkpoint runs/aurora_100m_qat/best.pt \
  --output model.aurora-tq \
  --calibrate --calibration-steps 4
```

Калибровка offline минимизирует ошибку реконструкции весов в каждой группе: поочерёдно пересчитываются ternary-уровни и оптимальный по least squares масштаб `sum(x*q)/sum(q*q)`, сохраняется лучший вариант. Она не ухудшает baseline `mean(|w|)` и не меняет формат архива: масштабы по-прежнему хранятся как FP16/FP32. Это калибровка по весам, а не по активациям; отдельный activation-calibration ещё нужен.

`load_tq_model` деквантует ternary-веса в обычную dense-модель для проверки и совместимости; memory-mapped runtime и быстрый ternary matmul пока не реализованы.

### Transformer baseline

```bash
python -m aurora.training.pretrain --config configs/transformer_100m.yaml
```

Decoder-only Transformer с RoPE, SwiGLU и KV-cache реализован для сравнения при равных данных, токенах и compute. Его total parameters (106.28M) близки к AURORA-100M (105.73M), но активные параметры отличаются: у AURORA 49.10M active из-за MoE. Любое сравнение должно явно указывать, используется total или active compute. Baseline использует тот же trainer, eval harness и формат чекпоинтов.

### Авто-resume и проверка чекпоинтов

```bash
python -m aurora.training.pretrain \
  --config configs/aurora_100m.yaml \
  --auto-resume
```

`--auto-resume` подхватывает `latest.pt` из `output_dir`, если он существует. При загрузке любой чекпоинт проверяется на обязательные ключи, версию формата и NaN/Inf в весах.

### Провенанс экспериментов

Каждый чекпоинт формата v2 содержит `run_metadata`: `run_id`, время старта и resume, git commit/branch/dirty, окружение (Python, платформа, Torch, NumPy, CUDA/cuDNN, устройство) и provenance данных — пути, split, SHA-256, `manifest.dataset_version` и общий dataset fingerprint. Чекпоинт также хранит последние train/validation-метрики и путь к JSONL-логу.

Trainer пишет append-only `runs/<run>/metrics.jsonl` со строками `train`, `validation` и `resume`; каждая строка содержит `run_id`, шаг, timestamp и метрики. При resume `run_id` сохраняется, а fingerprint данных сравнивается с сохранённым: изменение содержимого `.bin`, mixture-весов, split или версии manifest приводит к ошибке `Dataset provenance mismatch`.

Eval-отчёт (`reports/smoke_eval.json`) теперь включает `run_metadata` чекпоинта, git/окружение самой оценки и `dataset_version` каждого проверенного файла. Формат описан в `docs/formats.md`.

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
- Атомарные чекпоинты, resume, авто-resume, best/latest, проверка целостности тензоров и run metadata (git, окружение, dataset fingerprint).
- JSONL-лог train/validation-метрик и сводка метрик в чекпоинте.
- `L_memory` и `L_consistency` для ассоциативной памяти.
- Validation loss и perplexity.
- SentencePiece и byte-level tokenizer; `aurora-tokenizer-stats` для fertility/compression.
- Потоковое курирование корпуса: exact/near dedup, фильтры качества, PII drop/redact, JSON-отчёт и `dataset_version`.
- Потоковая подготовка данных с SHA-256 manifest и content-addressed `dataset_version`.
- Автоматическая проверка manifest для train/val/mixture-путей на запуске и при resume.
- Генерация data card из manifest с проверкой checksums (`aurora-data-card`).
- Memmap-датасет для токенизированных корпусов и смешивание доменов с весами.
- Transformer baseline с RoPE и KV-cache, совместимый с trainer/inference/eval.
- Единый eval harness с per-domain perplexity, dataset_version и git/окружением в JSON-отчёте.
- Ternary-квантование, упаковка 2-bit, QAT через straight-through estimator и версионированный AURORA-TQ архив.
- Offline-калибровка групповых масштабов по минимальной ошибке реконструкции (`--calibrate`).
- Загрузка чекпоинтов для инференса с автоматическим включением ternary-режима.
- Генерация с temperature, top-k, top-p и repetition penalty; батчевый и streaming-режимы.
- HTTP API для генерации (`aurora-serve`) и защита от слишком больших запросов.
- Gradient checkpointing, early stopping и остановка по token budget.
- Автоматический BF16/FP16 autocast на CUDA при генерации и оценке perplexity.
- CI (lint, mypy, tests, end-to-end smoke на чистом runner), `Makefile`, benchmark throughput/RAM/latency.
- 130 тестов: причинность, состояния, память, маршрутизация, квантование, обучение, resume, провенанс, чекпоинты, gradient checkpointing, mixture, document split, HTTP API, AURORA-TQ, data card, курирование корпуса, baseline и инференс.

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
│   │   ├── model.py
│   │   ├── transformer.py
│   │   ├── output.py
│   │   └── factory.py
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   ├── tokenizer/
│   │   ├── tokenizer.py
│   │   ├── train_tokenizer.py
│   │   └── stats.py
│   ├── data/
│   │   ├── curate.py
│   │   ├── prepare.py
│   │   ├── dataset.py
│   │   ├── mixture.py
│   │   ├── provenance.py
│   │   └── version.py
│   ├── quantization/
│   │   ├── ternary.py
│   │   ├── qat.py
│   │   └── tq.py
│   ├── benchmarks/
│   │   ├── perplexity.py
│   │   └── evaluate.py
│   ├── config.py
│   ├── generation.py
│   ├── inference.py
│   ├── provenance.py
│   ├── server.py
│   └── utils.py
├── configs/
├── docs/
│   ├── benchmarks.md
│   ├── formats.md
│   └── adr/
├── scripts/
│   ├── benchmark.py
│   ├── count_parameters.py
│   └── profile_components.py
├── tests/
├── .github/workflows/ci.yml
├── Makefile
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

- Рекуррентное ядро и память обрабатывают токены последовательно. Это упрощает верификацию, но ограничивает скорость обучения. Следующий шаг — chunkwise parallel scan и оптимизированные ядра.
- `L_memory` и `L_consistency` реализованы, но их коэффициенты не подобраны, ablation не проведён.
- Transformer baseline реализован, но честное сравнение на реальном корпусе при равных данных и compute ещё не проводилось.
- Все замеры throughput/latency/памяти сделаны на CPU. CUDA AMP и `torch.compile` в этом окружении не проверены.
- 80B-конфигурация не обучалась и не запускалась. Это расчётная конфигурация.
- Экспертная система распределена по слоям; стратегия разделяемых экспертов для 80B требует отдельных экспериментов.
- Курирование корпуса реализовано (`aurora-curate`), но на реальном корпусе ещё не прогонялось; точный и near-dedup индексы хранятся в RAM и не рассчитаны на миллиарды документов без шардирования. Data mixture поддерживается через `train_paths`/`train_weights`, подготовка данных потоковая.
- Нет распределённого обучения и C++/CUDA runtime. AURORA-TQ v1 реализован, но быстрый ternary matmul и memory-mapped загрузка экспертов ещё не сделаны.
- Пропускная способность RAM и предсказательная загрузка экспертов для 8 GB VRAM ещё не реализованы.

---

## Документация

- `docs/benchmarks.md` — замеры и команды воспроизведения.
- `docs/formats.md` — формат чекпоинтов, manifest и AURORA-TQ.
- `docs/adr/` — architecture decision records.

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
