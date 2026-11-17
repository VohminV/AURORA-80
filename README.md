# AURORA-80

<p align="center">
  <img src="logo.jpg" alt="AURORA-80" width="720">
</p>

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
| `configs/aurora_10m.yaml` | 10.24M | 8.76M | Нижняя точка scaling series |
| `configs/aurora_30m.yaml` | 30.71M | 20.09M | Средняя точка scaling series |
| `configs/aurora_100m.yaml` | 105.73M | 49.10M | Первый прототип |
| `configs/transformer_100m.yaml` | 106.28M | 106.28M | Transformer baseline |
| `configs/aurora_300m.yaml` | 300.62M | 94.18M | Верхняя точка scaling series |
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

### Docker

CPU-образ собирается из `python:3.12-slim`; по умолчанию entrypoint — `aurora-train`:

```bash
make docker-build
docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/runs:/workspace/runs" \
  aurora-80:cpu --config configs/aurora_100m.yaml
```

Inference запускается через переопределение entrypoint:

```bash
docker run --rm -v "$PWD/runs:/workspace/runs" \
  --entrypoint aurora-generate aurora-80:cpu \
  --checkpoint runs/smoke/best.pt \
  --prompt "AURORA" --max-new-tokens 16
```

Для CUDA-сборки передайте индекс PyTorch с CUDA-колёсами и запускайте с `--gpus all`:

```bash
docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  -t aurora-80:cu128 .
docker run --rm --gpus all -v "$PWD/data:/workspace/data" \
  -v "$PWD/runs:/workspace/runs" aurora-80:cu128 \
  --config configs/aurora_100m.yaml --device cuda
```

Контейнер работает от непривилегированного пользователя `aurora` (UID 10001); каталоги `data`, `runs` и `reports` должны быть доступны ему на запись. Данные и чекпоинты не входят в образ — монтируйте их через `-v`.

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

### 6. Проверка длинного контекста

```bash
python -m aurora.benchmarks.state_stability \
  --checkpoint runs/smoke/best.pt \
  --lengths 4096 8192 16384 32768 \
  --chunk-size 256 \
  --output reports/state_stability.json
```

Последовательность обрабатывается чанками с продолжением состояния, для каждой длины считается loss/perplexity, скорость, финитность, норма и максимальное абсолютное значение состояния, рост нормы и loss первого/последнего чанка. `--reset-interval N` сбрасывает состояние каждые N токенов, что позволяет сравнить единое состояние с документными границами. Для воспроизводимости можно передать `--data tokens.bin`; без него токены детерминированно генерируются по seed. Замер на smoke-модели проверяет численную устойчивость, но не качество длинного контекста обученной модели.

### 7. Абляция компонентов

```bash
python -m aurora.benchmarks.ablation \
  --config configs/smoke.yaml \
  --variants core core_memory full \
  --output reports/smoke_ablation.json
```

Варианты обучаются с нуля на одних данных, seed и token budget: `core` без памяти и MoE, `core_memory` с памятью, `full` с памятью и MoE. Отчёт содержит параметры (total/active), loss/perplexity, скорость, run id и чекпоинты, а также `loss_delta_vs_core` и `perplexity_ratio_vs_core`. Отключённые компоненты полностью удаляют свои параметры, состояние и лоссы. Smoke-прогон проверяет механику абляции; выводы о вкладе компонентов требуют обученного 100M-чекпоинта и нескольких seed.

Для model-size осей можно передать несколько значений `--n-levels`, `--d-state`, `--memory-heads`, `--n-experts` и `--top-k`:

```bash
python -m aurora.benchmarks.ablation \
  --config configs/smoke.yaml \
  --variants core full \
  --n-levels 1 2 \
  --d-state 8 16 \
  --n-experts 2 4 \
  --top-k 1 2 \
  --max-steps 10 \
  --output reports/smoke_axis_ablation.json
```

Строится декартово произведение значений в порядке осей. Оси, не относящиеся к пресету, фиксируются на базовом значении (`memory_heads` для `core`, `n_experts`/`top_k` для `core` и `core_memory`). Вариант с базовыми значениями сохраняет имя пресета, остальные получают суффиксы вида `_n_levels1_d_state8`; `--max-variants` (по умолчанию 128) ограничивает размер сетки, а невалидные комбинации вроде `top_k > n_experts` отклоняются до запуска. В отчёте у каждого варианта есть `overrides` и эффективные значения `model`.

### 8. Zero-shot multiple-choice оценка

```bash
python -m aurora.benchmarks.zero_shot \
  --checkpoint runs/aurora_100m/best.pt \
  --task mmlu=benchmarks/mmlu.jsonl \
  --task hellaswag=benchmarks/hellaswag.jsonl \
  --tokenizer sentencepiece \
  --tokenizer-model tokenizer/aurora_64k.model \
  --max-length 2048 \
  --normalize-by-length \
  --output reports/zero_shot.json \
  --predictions reports/zero_shot_predictions.jsonl
```

Каждая задача — JSONL с полями `context`, `choices` и `answer` (индекс правильного варианта): `{"context": "...", "choices": ["...", "..."], "answer": 1}`. Harness считает сумму log-prob продолжения каждого варианта после общего контекста и выбирает максимум; `--normalize-by-length` переходит к средней log-prob. `--limit` ограничивает число примеров, `--predictions` пишет построчные предсказания для анализа ошибок. Датасеты нужно предварительно привести к этому формату; harness не содержит загрузчиков MMLU/HellaSwag/ARC/WinoGrande и не выполняет few-shot сборку промптов. Оценка идёт последовательно по вариантам, без батчинга, поэтому на больших бенчмарках она медленная.

### 9. Сводный отчёт по запускам

```bash
python -m aurora.report \
  --report seed0=reports/eval_seed0.json \
  --report seed1=reports/eval_seed1.json \
  --report seed0_zeroshot=reports/zero_shot_seed0.json \
  --output reports/results.md \
  --json reports/results.json
```

Агрегатор принимает eval-отчёты (`results.<domain>.perplexity`) и zero-shot-отчёты (`tasks.<task>.accuracy`), считает mean, выборочное std (ddof=1), min/max и число запусков по каждой задаче. В Markdown-таблице также перечислены run id, git commit, шаг чекпоинта и версии датасетов. Инструмент не проверяет, что запуски используют одинаковые данные, compute, токенизатор и seed, и не выполняет статистических тестов: при одном отчёте std равен 0.

### 10. Реестр экспериментов

```bash
python -m aurora.registry \
  --runs runs \
  --reports reports \
  --verify \
  --output reports/registry.json \
  --markdown reports/registry.md
```

`aurora-registry` рекурсивно собирает `*.pt`-чекпоинты и `*.json`-отчёты, группирует их по `run_metadata.run_id` и пишет JSON/Markdown с git commit, шагом, seed, версиями датасетов и сводкой метрик. Отчёты распознаются как eval, zero-shot, quantization, ablation, state stability, sweep, scaling, сводный отчёт или dataset manifest; неизвестные JSON помечаются `unknown`. Метаданные чекпоинтов читаются через `weights_only=True` и `mmap`, поэтому большие веса не обязаны полностью помещаться в RAM. `--verify` дополнительно считает SHA-256 всех артефактов и читает их целиком. Битые файлы попадают в `errors`, но не прерывают сбор. Реестр не проверяет finite-тензоры и качество моделей; для этого есть `aurora-evaluate` и валидация чекпоинтов при загрузке.

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

### Статистика корпуса

```bash
python -m aurora.data.corpus_stats \
  --input data/curated/corpus_ru.txt data/curated/corpus_en.txt data/curated/corpus_code.jsonl \
  --domain-field source \
  --output reports/corpus_stats.json
```

Инструмент потоково считает документы, символы, слова, строки, долю code-строк и распределение письменностей по доменам. Домен берётся из JSONL-поля `--domain-field` или из имени файла. `language_hint` (`cyrillic`/`latin`/`cjk`/`code`/`mixed`) — эвристика по письменности и доле code-строк, а не language identification; токены не считаются, потому что зависят от токенизатора (для fertility используйте `aurora-tokenizer-stats`). Отчёт описан в `docs/formats.md`.

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

### Model card

```bash
python -m aurora.model_card \
  --checkpoint runs/aurora_100m/best.pt \
  --evaluation reports/aurora_100m_eval.json \
  --output reports/aurora_100m_MODEL_CARD.md \
  --name "AURORA-100M" \
  --license "TBD"
```

Карточка собирается из конфига чекпоинта, `run_metadata`, метрик и eval-отчёта: параметры total/active, архитектура, данные и их checksums, git/окружение, per-domain perplexity, ограничения и условия использования. SHA-256 самого чекпоинта включается в карточку. Если обученного чекпоинта ещё нет, можно создать черновик из конфига:

```bash
python -m aurora.model_card \
  --config configs/aurora_100m.yaml \
  --output docs/model_card_100m.md \
  --name "AURORA-100M (prototype draft)"
```

Генератор не подтверждает лицензии данных, отсутствие contamination и safety-свойства; эти проверки должны быть задокументированы отдельно. Черновик AURORA-100M лежит в `docs/model_card_100m.md`.

### Проверка benchmark contamination

```bash
python -m aurora.data.contamination \
  --corpus data/curated/corpus.txt \
  --benchmark benchmarks/mmlu.jsonl benchmarks/hellaswag.txt \
  --json-field text \
  --ngram-size 13 \
  --output reports/contamination.json \
  --fail-on-match
```

Инструмент нормализует Unicode, приводит текст к нижнему регистру и ищет пересечения word n-грамм корпуса с каждым benchmark-документом. В отчёте — доля benchmark-документов с пересечениями, число совпавших n-грамм, примеры с путями и номерами документов, а также ограничения метода. Память пропорциональна размеру бенчмарков, а не корпуса: корпус читается потоково. `--fail-on-match` завершает процесс кодом 1, что удобно для CI. Совпадение n-грамм — сигнал для ручного разбора, а не доказательство contamination; перевод, перефразирование и смена токенизации могут обходить проверку. Формат отчёта описан в `docs/formats.md`.

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

При `training.document_reset: true` trainer строит reset-маску по EOS (`training.eos_token_id`, по умолчанию 2) и сбрасывает состояние recurrent core и associative memory перед первым токеном нового документа. Eval, perplexity и quantization benchmark используют ту же маску из конфига чекпоинта. Reset поддерживается только для `architecture: aurora`; для Transformer baseline trainer завершается с ошибкой. Включайте флаг, если `prepare.py` добавлял EOS после каждого документа.

### Hyperparameter sweep

```bash
python -m aurora.sweep \
  --spec configs/sweeps/smoke_lr.yaml \
  --output reports/smoke_sweep.json \
  --markdown reports/smoke_sweep.md
```

Sweep-спека задаёт базовый конфиг, сетку параметров, seeds и token budget:

```yaml
base: configs/smoke.yaml
output_dir: runs/sweeps/smoke_lr
max_steps: 25
seeds: [1, 2]
parameters:
  training.learning_rate: [0.0001, 0.0003]
  training.weight_decay: [0.0, 0.1]
```

`aurora-sweep` строит декартово произведение параметров, для каждого trial пишет собственный `config.yaml` в `output_dir/<trial>/` и обучает модель на одних и тех же train/val-файлах с одинаковым token budget. Сетка ограничена `--max-trials` (по умолчанию 128). Тюнить можно любые поля `model.*` и оптимизационные поля `training.learning_rate`, `weight_decay`, `beta1`, `beta2`, `adam_eps`, `warmup_steps`, `min_lr_ratio`, `grad_clip`; seeds задаются отдельным списком. `--dry-run` только генерирует конфиги без обучения, `--train-path`, `--val-path`, `--device` и `--max-steps` переопределяют базу. Отчёт содержит каждый trial с loss/perplexity, tokens/s, параметрами и run id, а также конфигурации, ранжированные по среднему validation loss по seeds. Ошибка отдельного trial записывается в `failed` и не прерывает sweep. Сравнение не проверяет равенство compute при изменении `model.*`: такие trials меняют число параметров и стоимость шага.

### Scaling law fit

```bash
python -m aurora.scaling \
  --point 10m=runs/aurora_10m/best.pt \
  --point 30m=runs/aurora_30m/best.pt \
  --point 100m=runs/aurora_100m/best.pt \
  --holdout 300m \
  --compute-budget 1e20 --compute-budget 1e21 \
  --output reports/scaling.json \
  --markdown reports/scaling.md
```

`aurora-scaling` принимает чекпоинты (`--point NAME=PATH.pt`) и/или JSON с явными точками (`--points`). Из чекпоинта берутся validation loss из `metrics.last_validation.loss` (fallback `best_val`), `metrics.tokens_seen` (fallback `step * batch_size * grad_accum_steps * seq_len`) и число параметров, посчитанное по конфигу на meta-устройстве. Для каждой оси — `parameters`, `active_parameters`, `tokens`, `compute = 6*N*D` и `active_compute` — фитится `L = E + A * x^-alpha`; при ≥5 точках дополнительно фитится `L = E + A*N^-alpha + B*D^-beta`, а `--compute-budget` оценивает оптимальные `N` и `D` при `6*N*D = C`. Авто-holdout включается при ≥5 точках и оставляет ≥4 для фита; явный `--holdout` пропускается, если для фита остаётся меньше четырёх точек. Отчёт содержит точки, параметры фитов, ошибку holdout, bivariate-фит с conditioning и compute-optimal оценки, а также список ограничений.

Точки можно задать вручную:

```json
{
  "points": [
    {
      "name": "30m",
      "total_parameters": 30710704,
      "active_parameters": 20093872,
      "tokens_seen": 614400000,
      "loss": 3.12
    }
  ]
}
```

Для устойчивого фита нужно минимум четыре точки (пять для bivariate и авто-holdout), один корпус, токенизатор, `seq_len` и протокол валидации. `compute` считается по полному числу параметров, поэтому для MoE-моделей он завышен относительно training FLOPs по активным параметрам. Реальная серия 10M–300M ещё не обучена: инструмент и синтетические тесты готовы, эмпирические fit/holdout появятся после запусков.

### Возобновление обучения

```bash
python -m aurora.training.pretrain \
  --config configs/aurora_100m.yaml \
  --resume runs/aurora_100m/latest.pt
```

### Бэкапы чекпоинтов и алерты

```yaml
training:
  backup_interval: 1000
  backup_count: 3
  backup_dir: runs/aurora_100m_backups
  degradation_patience: 3
```

При сохранении `best.pt` или `latest.pt` на шаге, кратном `backup_interval`, копия пишется в `backup_dir` (по умолчанию `<output_dir>/backups`) как `<best|latest>_step_<step>.pt`. Для каждого префикса хранятся только последние `backup_count` копий, старые удаляются. Копирование атомарное; ошибка бэкапа не прерывает обучение. Флаги `--backup-interval`, `--backup-count`, `--backup-dir` переопределяют конфиг.

Алерты отправляются POST-запросом с JSON на webhook из переменной окружения `AURORA_ALERT_WEBHOOK` (URL не попадает в конфиг и чекпоинт):

```bash
AURORA_ALERT_WEBHOOK=https://example.com/aurora-alerts \
python -m aurora.training.pretrain --config configs/aurora_100m.yaml
```

События: `validation_degradation` (после `degradation_patience` validation-замеров без улучшения, один раз за серию), `early_stopping`, `non_finite_loss` и `non_finite_grad_norm`. Payload содержит `event`, `created_utc`, `run_id`, `step` и детали события. Сеть и ошибки доставки не останавливают обучение; при недоступном webhook пишется предупреждение. Алерты не заменяют мониторинг и не ретраятся.

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

### Сравнение квантизации

```bash
python -m aurora.benchmarks.quantization \
  --checkpoint runs/aurora_100m/best.pt \
  --data data/processed/val.bin \
  --variants fp32 fp16 ternary ternary_calibrated int8 int4 \
  --group-size 128 \
  --min-features 256 \
  --output reports/quantization_compare.json
```

Все варианты оцениваются на одном чекпоинте и одном наборе данных. FP16, ternary (с калибровкой и без), INT8 и INT4 — weight-only fake quantization: веса подходящих `nn.Linear` заменяются квантованными и деквантованными значениями, остальные слои не меняются. Отчёт содержит loss/perplexity, прирост perplexity относительно FP32, weight MSE, число квантованных параметров и оценку `bit/weight` с учётом FP16-масштабов. Это сравнение качества и оценка размера, а не замер скорости: быстрых INT/ternary-ядер пока нет. QAT-чекпоинты не принимаются; для них используйте `aurora-tq export`.

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

### Логирование

```bash
pip install -e ".[logging]"
python -m aurora.training.pretrain --config configs/smoke.yaml --tensorboard
python -m aurora.training.pretrain --config configs/smoke.yaml \
  --wandb --wandb-project aurora-80 --wandb-mode offline
```

TensorBoard и W&B включаются независимо; если пакет не установлен или `wandb.init` не удался, обучение продолжается с предупреждением. W&B получает полный конфиг, шаг и метрики `train/*`/`validation/*`, а при resume использует сохранённый `run_id` как id рана. Режим `--wandb-mode offline` не требует сети.

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
- Потоковая статистика корпуса: домены, письменности, code-строки и `language_hint` (`aurora-corpus-stats`).
- Потоковая подготовка данных с SHA-256 manifest и content-addressed `dataset_version`.
- Автоматическая проверка manifest для train/val/mixture-путей на запуске и при resume.
- Генерация data card из manifest с проверкой checksums (`aurora-data-card`).
- Генерация model card из конфига/чекпоинта, провенанса и eval-отчёта (`aurora-model-card`).
- Потоковая проверка benchmark contamination по word n-граммам (`aurora-contamination`).
- Memmap-датасет для токенизированных корпусов и смешивание доменов с весами.
- Transformer baseline с RoPE и KV-cache, совместимый с trainer/inference/eval.
- Единый eval harness с per-domain perplexity, dataset_version и git/окружением в JSON-отчёте.
- Zero-shot multiple-choice harness с log-prob скорингом, per-task accuracy и JSONL-предсказаниями (`aurora-zero-shot`).
- Агрегация eval/zero-shot отчётов с mean/std по запускам и Markdown-таблицей (`aurora-report`).
- Ротация бэкапов best/latest-чекпоинтов и необязательные webhook-алерты при деградации, early stopping и non-finite loss/grad.
- Реестр экспериментов и артефактов с группировкой по run id и опциональной проверкой SHA-256 (`aurora-registry`).
- Ternary-квантование, упаковка 2-bit, QAT через straight-through estimator и версионированный AURORA-TQ архив.
- Offline-калибровка групповых масштабов по минимальной ошибке реконструкции (`--calibrate`).
- Сравнение FP32/FP16/ternary/INT8/INT4 на одном чекпоинте и датасете (`aurora-quant-bench`).
- Проверка стабильности состояния на 4k–32k с опциональным reset (`aurora-state-check`).
- State reset на границах документов в обучении и оценке (`training.document_reset`, EOS-маска для core и memory).
- Абляция `core`/`core_memory`/`full` и осей `n_levels`/`d_state`/`memory_heads`/`n_experts`/`top_k` с равным token budget (`aurora-ablate`).
- Hyperparameter sweep runner с генерацией конфигов, seeds, запуском и сводкой по validation loss (`aurora-sweep`).
- Конфиги 10M/30M/300M и fitting scaling law по loss от parameters/tokens/compute с holdout и compute-optimal оценкой (`aurora-scaling`).
- Загрузка чекпоинтов для инференса с автоматическим включением ternary-режима.
- Генерация с temperature, top-k, top-p и repetition penalty; батчевый и streaming-режимы.
- HTTP API для генерации (`aurora-serve`) и защита от слишком больших запросов.
- Gradient checkpointing, early stopping и остановка по token budget.
- Автоматический BF16/FP16 autocast на CUDA при генерации и оценке perplexity.
- CUDA AMP (BF16, FP16+GradScaler) и `torch.compile` (`aot_eager`, `inductor`) проверены на 100M; CUDA-замеры — в `docs/benchmarks.md`.
- CI (lint, mypy, tests, end-to-end smoke на чистом runner), `Makefile`, benchmark throughput/RAM/latency.
- 229 тестов: причинность, состояния, document reset, память, маршрутизация, квантование, сравнение квантизации, длинный контекст, абляция компонентов и model-size осей, hyperparameter sweep, scaling law fit, zero-shot оценка, агрегация результатов, реестр экспериментов, обучение, resume, провенанс, чекпоинты, бэкапы и алерты, gradient checkpointing, mixture, document split, HTTP API, AURORA-TQ, data card, model card, contamination, статистика корпуса, W&B tracker, курирование корпуса, baseline и инференс.

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
│   │   ├── alerts.py
│   │   ├── tracker.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   ├── tokenizer/
│   │   ├── tokenizer.py
│   │   ├── train_tokenizer.py
│   │   └── stats.py
│   ├── data/
│   │   ├── contamination.py
│   │   ├── corpus_stats.py
│   │   ├── curate.py
│   │   ├── prepare.py
│   │   ├── dataset.py
│   │   ├── mixture.py
│   │   ├── provenance.py
│   │   └── version.py
│   ├── quantization/
│   │   ├── ternary.py
│   │   ├── baselines.py
│   │   ├── qat.py
│   │   └── tq.py
│   ├── benchmarks/
│   │   ├── perplexity.py
│   │   ├── quantization.py
│   │   ├── state_stability.py
│   │   ├── ablation.py
│   │   ├── zero_shot.py
│   │   └── evaluate.py
│   ├── config.py
│   ├── generation.py
│   ├── inference.py
│   ├── model_card.py
│   ├── provenance.py
│   ├── registry.py
│   ├── report.py
│   ├── server.py
│   ├── scaling.py
│   ├── sweep.py
│   └── utils.py
├── configs/
├── docs/
│   ├── benchmarks.md
│   ├── formats.md
│   ├── model_card_100m.md
│   └── adr/
├── scripts/
│   ├── benchmark.py
│   ├── count_parameters.py
│   └── profile_components.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
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
- `L_memory` и `L_consistency` реализованы, но их коэффициенты не подобраны. Smoke-абляция `core`/`core_memory`/`full` есть, вклад компонентов на обученном 100M ещё не измерен.
- Document reset срабатывает только по явным EOS-маркерам (`training.document_reset`), а loss на переходе EOS → первый токен нового документа не маскируется. Transformer baseline reset не поддерживает; learned initial state по-прежнему не выбран (ADR 0004).
- Transformer baseline реализован, но честное сравнение на реальном корпусе при равных данных и compute ещё не проводилось.
- CUDA AMP и `torch.compile` проверены на RTX 3050: eager CUDA медленнее CPU из-за последовательного ядра, `inductor` ускоряет batched inference, но MoE упирается в recompile limit, поэтому compiled training/generation медленнее eager. CUDA graphs не проверены. Основной `.venv` остаётся CPU-only; CUDA-замеры выполнены в отдельном `.venv-cuda`.
- 80B-конфигурация не обучалась и не запускалась. Это расчётная конфигурация.
- Экспертная система распределена по слоям; стратегия разделяемых экспертов для 80B требует отдельных экспериментов.
- Курирование корпуса реализовано (`aurora-curate`), но на реальном корпусе ещё не прогонялось; точный и near-dedup индексы хранятся в RAM и не рассчитаны на миллиарды документов без шардирования. Data mixture поддерживается через `train_paths`/`train_weights`, подготовка данных потоковая. `aurora-corpus-stats` даёт эвристическое распределение языков/доменов, но не заменяет language identification и не считает токены.
- Нет распределённого обучения и C++/CUDA runtime. AURORA-TQ v1 реализован, но быстрый ternary matmul и memory-mapped загрузка экспертов ещё не сделаны.
- Пропускная способность RAM и предсказательная загрузка экспертов для 8 GB VRAM ещё не реализованы.
- `aurora-sweep` реализован как инструмент, но sweep LR/betas/warmup/weight decay на реальном корпусе ещё не проведён; smoke-sweep проверяет только механику генерации конфигов, запуска и сводки.
- Серия 10M/30M/100M/300M не обучена: конфиги и `aurora-scaling` готовы, но эмпирический scaling law, holdout на 300M и compute-optimal budget ещё не подтверждены.

---

## Документация

- `docs/benchmarks.md` — замеры и команды воспроизведения.
- `docs/formats.md` — формат чекпоинтов, manifest, model card, отчётов corpus stats/contamination/zero-shot/сводного/реестра/sweep/scaling и AURORA-TQ.
- `docs/model_card_100m.md` — черновик model card для конфигурации AURORA-100M.
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
