# AURORA-80 — дорожная карта исследования

> Концепция: проверить, может ли сочетание многоуровневого рекуррентного ядра,
> ассоциативной памяти с delta rule и разреженной маршрутизации экспертов дать
> конкурентное качество языкового моделирования при меньшем active compute и
> inference в целевых 8 GB VRAM.
>
> Главное правило: этап закрывается только когда выполнены все его `P0` и пройден
> gate. Любое сравнение с другой моделью допустимо только при равных данных,
> токенах, compute и едином протоколе оценки. Отрицательный результат фиксируется
> так же честно, как положительный.

## Легенда

- `[x]` — сделано и проверено;
- `[ ]` — не сделано;
- `P0` — блокирует переход к следующему этапу;
- `P1` — важно для качества, масштабирования или воспроизводимости;
- `P2` — улучшение и полировка;
- «инструмент готов» — код, тесты и документация есть, но реальный прогон на
  данных/чекпоинте ещё не выполнен.

---

## 0. Срез проекта

### 0.1 Что уже готово

**Архитектура и обучение**

- [x] Многоуровневое рекуррентное ядро с межуровневым взаимодействием.
- [x] Ассоциативная память с delta rule, decay/write gates.
- [x] MoE с Top-K, load-balancing loss и z-loss.
- [x] Residual integration, LayerScale, RMSNorm, SwiGLU, tied embeddings.
- [x] `LanguageHead`, корректная инициализация и причинность.
- [x] `L_memory` и `L_consistency`.
- [x] Transformer baseline того же размера с RoPE и KV-cache.
- [x] AdamW с раздельным weight decay, cosine scheduler с warmup.
- [x] BF16/FP16 autocast, GradScaler, gradient accumulation и clipping.
- [x] Пропуск non-finite шагов, early stopping и остановка по token budget.
- [x] Gradient checkpointing и проверенный `torch.compile`
  (`aot_eager`, `inductor`; MoE recompiles остаются).
- [x] Checkpoint save/load, atomic replace, resume, auto-resume, best/latest.
- [x] Проверка чекпоинтов на обязательные ключи, версию и non-finite тензоры.
- [x] Ротация бэкапов best/latest и webhook-алерты при деградации.
- [x] Опциональные TensorBoard и W&B (online/offline, graceful fallback).
- [x] Hyperparameter sweep runner: сетка параметров, seeds, генерация
  конфигов, запуск и сводка (`aurora-sweep`).

**Данные и токенизация**

- [x] Byte-level tokenizer и SentencePiece tokenizer.
- [x] Потоковая подготовка `.bin` и memmap-датасет.
- [x] Manifest с SHA-256, `dataset_version` и проверкой при запуске/resume.
- [x] Train/val/test split без пересечения документов.
- [x] Data mixture с весами доменов.
- [x] Потоковое курирование: exact/near dedup, качество, PII drop/redact.
- [x] Потоковая статистика корпуса: домены, письменности, code-доли
  (`aurora-corpus-stats`).
- [x] Проверка benchmark contamination по word n-граммам
  (`aurora-contamination`).
- [x] `aurora-tokenizer-stats` для fertility/compression.
- [x] Генератор data card с проверкой checksums (`aurora-data-card`).

**QAT и формат весов**

- [x] Ternary quantization, STE, packing/unpacking.
- [x] `TernaryLinear`, конвертация модели и QAT resume/inference.
- [x] Offline-калибровка групповых ternary-масштабов (`--calibrate`).
- [x] Сравнение FP32/FP16/ternary/INT8/INT4 на одном чекпоинте
  (`aurora-quant-bench`).
- [x] Версионированный формат AURORA-TQ v1 (`save_tq`/`load_tq`, `aurora-tq`).

**Оценка и inference**

- [x] PyTorch reference inference, state continuation без KV-cache.
- [x] Temperature, top-k, top-p, repetition penalty.
- [x] Батчевая и streaming-генерация.
- [x] HTTP API (`aurora-serve`, localhost по умолчанию).
- [x] Perplexity и единый eval harness с per-domain отчётом.
- [x] Zero-shot multiple-choice harness (`aurora-zero-shot`).
- [x] Агрегация результатов с mean/std по запускам (`aurora-report`).
- [x] Реестр экспериментов и артефактов (`aurora-registry`).
- [x] Проверка стабильности состояния 4k/8k/16k/32k (`aurora-state-check`).
- [x] Абляция `core`/`core_memory`/`full` (`aurora-ablate`).
- [x] Генератор model card и черновик `docs/model_card_100m.md`.

**Инженерия и воспроизводимость**

- [x] Пакет `aurora`, entry points, строгая валидация YAML.
- [x] Seed, восстановление RNG, провенанс эксперимента.
- [x] Run metadata в чекпоинте: git, окружение, seed, версия данных.
- [x] JSONL-метрики и сводка метрик внутри чекпоинта.
- [x] CI: lint, mypy, tests и end-to-end smoke на чистом runner.
- [x] Docker CPU/CUDA, `Makefile`.
- [x] ADR и документация форматов.
- [x] 218 тестов; `ruff`, `mypy` и format-check проходят.

### 0.2 Чего ещё нет

- [ ] Реального корпуса RU/EN/code и обученного токенизатора 64k.
- [ ] Обученного AURORA-100M и честного сравнения с Transformer baseline.
- [ ] Публичных бенчмарков и отчёта с результатами нескольких seed.
- [ ] Distributed training, FSDP/TP/EP и sharded checkpoints.
- [ ] Chunkwise parallel scan, fused kernels и быстрого ternary matmul.
- [ ] C++/CUDA runtime, memory-mapped экспертов и целевых 8 GB VRAM.
- [ ] SFT, alignment и safety-оценок.
- [ ] Определённых лицензий кода, весов и данных.
- [ ] Финальных model card и data card обученной модели.
- [ ] Доказательств превосходства над DeepSeek или любой другой моделью.

### 0.3 Зафиксированные ориентиры

| Конфигурация | Total params | Active params |
| --- | ---: | ---: |
| AURORA-100M | 105.73M | 49.10M |
| Transformer-100M baseline | 106.28M | 106.28M |
| AURORA-1B | 929.49M | 325.51M |
| AURORA-80B | 80.49B | 11.52B |

Smoke/CUDA-ориентиры: ~22 tok/s на 100M при обучении на RTX 3050; 2.7 GB
allocated / 3.2 GB reserved; gradient checkpointing даёт −24% VRAM и −41%
throughput; CUDA eager медленнее CPU из-за последовательного ядра.

### 0.4 Честные ограничения текущего среза

- Smoke-обучение и абляции проверяют механику, но не доказывают качество
  архитектуры.
- QAT и квантизационное сравнение не заменяют FP baseline на обученном 100M.
- Inference-замеры 8 GB VRAM относятся к dense PyTorch-пути, а не к будущему
  runtime.
- `aurora-zero-shot`, `aurora-report` и `aurora-registry` готовы как инструменты,
  но реальные бенчмарки и multi-seed отчёты ещё не получены.
- `aurora-sweep` готов как инструмент, но sweep-протокол LR/betas/warmup/weight
  decay на малом масштабе ещё не проведён и не зафиксирован.

---

## 1. M0 — Foundation: воспроизводимый прототип

**Статус: закрыт.**

- [x] `P0` Пакет, entry points, конфиги, CI, Docker, `Makefile`.
- [x] `P0` Архитектура AURORA и Transformer baseline.
- [x] `P0` Обучение, validation, checkpoints, resume, QAT.
- [x] `P0` Данные: prepare, manifest, versioning, mixture, document split.
- [x] `P0` Inference: generation, perplexity, HTTP API, batched/streaming.
- [x] `P0` 218 тестов, `ruff`, `mypy`, end-to-end smoke на чистом runner.
- [x] `P0` Проверены AMP, gradient checkpointing и `torch.compile` на CUDA.
- [x] `P0` Замерены throughput, память и latency прототипа (CPU и CUDA).

**Gate M0 пройден:** smoke-обучение воспроизводимо, чекпоинт можно сохранить,
загрузить и использовать, QAT обучается и запускается, CI зелёный.

---

## 2. M1 — Research baseline 100M (текущий этап)

**Цель:** получить воспроизводимый baseline AURORA-100M на реальном корпусе и
сравнить его с Transformer-аналогом при равных данных, токенах и compute.

### 2.1 Лицензии и корпус

- [ ] `P0` Определить лицензию кода, весов и правила использования.
- [ ] `P0` Собрать RU/EN/code-корпус не менее 2–5 млрд токенов после
  токенизации.
- [ ] `P0` Проверить лицензии источников и сохранить атрибуцию.
- [ ] `P0` Прогнать курирование: качество, exact/near dedup, PII.
- [ ] `P0` Прогнать contamination-check против планируемых бенчмарков.
- [ ] `P0` Зафиксировать immutable-версию датасета, SHA-256 и data card.
- [ ] `P0` Выделить отдельный test set, не использованный в обучении.
- [ ] `P1` Проанализировать распределение языков и доменов
  (`aurora-corpus-stats`).

### 2.2 Токенизатор

- [ ] `P0` Обучить SentencePiece-токенизатор 64k на реальном корпусе.
- [ ] `P0` Замерить fertility/compression для RU/EN/code
  (`aurora-tokenizer-stats`).
- [ ] `P0` Проверить бюджет embedding и совместимость с 100M-конфигом.
- [ ] `P1` Сравнить 32k/64k/128k по fertility и стоимости.
- [ ] `P1` Зафиксировать SHA-256 токенизатора в manifest.

### 2.3 Обучение

- [x] `P0` Реализовать hyperparameter sweep для LR/betas/warmup/weight decay
  (`aurora-sweep`: сетка, seeds, генерация конфигов, запуск и сводка).
- [ ] `P0` Провести sweep на малом масштабе и зафиксировать протокол.
- [ ] `P0` Обучить AURORA-100M на 1–2 млрд токенов.
- [ ] `P0` Повторить запуск минимум с двумя seed (или два независимых прогона).
- [ ] `P0` Логировать loss/LR/grad norm/tokens/s и проверять resume.
- [ ] `P1` Обучить Transformer-100M на тех же данных, токенах и compute.
- [ ] `P1` Сравнить с Mamba-2/RWKV-7 baseline, если доступны.
- [ ] `P1` Явно указывать total и active compute в отчётах.
- [ ] `P2` Curriculum learning и context extension.

### 2.4 Оценка

- [ ] `P0` Held-out perplexity на RU/EN/code.
- [ ] `P0` Zero-shot MMLU, HellaSwag, ARC, WinoGrande
  (harness `aurora-zero-shot` готов; нужны данные).
- [ ] `P0` RU-бенчмарки: ruMMLU, Russian SuperGLUE и аналоги.
- [ ] `P0` Code: HumanEval, MBPP.
- [ ] `P0` Отчёт с таблицей результатов и погрешностями
  (агрегатор `aurora-report` готов).
- [ ] `P0` Зафиксировать вывод: гипотеза подтверждается, опровергается или
  требует изменений.
- [ ] `P1` Анализ ошибок и типичных failure modes.
- [ ] `P1` Human evaluation для RU/EN.
- [ ] `P1` Проверить state stability на длинах обучения.

### Gate M1

- [ ] Loss стабильно снижается, нет деградации состояния.
- [ ] Модель превосходит random baseline.
- [ ] Есть честное сравнение с Transformer при равных параметрах/токенах/compute.
- [ ] Есть отчёт с mean/std минимум по двум запускам.
- [ ] Все эксперименты воспроизводятся по README.
- [ ] Принято решение о переходе к M2 или изменении архитектуры.

---

## 3. M2 — Абляции, scaling laws и длинный контекст

### 3.1 Архитектурные абляции

- [ ] `P0` Core only.
- [ ] `P0` Core + memory.
- [ ] `P0` Core + memory + MoE.
- [ ] `P0` Варианты residual integration и LayerScale.
- [ ] `P0` Вклад `L_memory` и `L_consistency` и их коэффициентов.
- [ ] `P0` Влияние `n_levels` и `d_state`.
- [ ] `P0` Влияние числа экспертов и `top_k`.
- [ ] `P1` Варианты decay/write gates.
- [ ] `P1` Варианты нормализации состояния.
- [ ] `P1` Shared experts vs independent experts.
- [ ] `P2` Варианты памяти: linear attention, delta-net, Titans-like.
- [ ] `P2` Expert choice routing.

Текущий `aurora-ablate` закрывает только `core`/`core_memory`/`full`; для
остальных пунктов его нужно расширить конфигурационными вариантами.

### 3.2 Scaling laws

- [x] `P0` Добавить конфиги 10M/30M/100M/300M.
- [ ] `P0` Прогнать серию 10M/30M/100M/300M на одном корпусе и протоколе.
- [ ] `P0` Fit scaling law по loss от compute/parameters/tokens на реальных запусках.
- [ ] `P0` Проверить предсказание на 300M.
- [ ] `P1` Оценить compute-optimal token budget на реальной серии (инструмент готов).
- [ ] `P1` Сравнить scaling curve с Transformer baseline.
- [x] `P1` Сделать fitting-инструмент с артефактом отчёта (`aurora-scaling`; проверен на синтетике).

### 3.3 Длинный контекст и состояние

- [ ] `P0` Обучение/оценка на 2k/4k/8k.
- [ ] `P0` Проверка деградации памяти и состояния.
- [ ] `P0` Решить и реализовать state reset на границах документов.
- [ ] `P1` Chunkwise parallel scan.
- [ ] `P1` Оценка long-context задач.
- [ ] `P1` Сравнить нулевое состояние, reset и learned initial state.
- [ ] `P2` Расширение до 32k+.

### Gate M2

- [ ] Выбрана финальная архитектура для 1B.
- [ ] Есть scaling law с приемлемой ошибкой и проверкой на 300M.
- [ ] Понятно, какие компоненты дают вклад, а какие можно удалить.
- [ ] Long-context поведение измерено и задокументировано.

---

## 4. M3 — Масштаб 1B

### 4.1 Инфраструктура

- [ ] `P0` Distributed Data Parallel.
- [ ] `P0` FSDP или sharded optimizer.
- [ ] `P0` Sharded checkpoints и быстрое восстановление.
- [ ] `P0` Expert parallelism для MoE.
- [ ] `P0` Мониторинг GPU/RAM/network.
- [ ] `P1` Mixed precision на реальном кластере.
- [ ] `P1` Gradient/activation checkpointing на масштабе.
- [ ] `P1` Отказоустойчивость и авто-resume на длинном прогоне.

### 4.2 Данные и обучение

- [ ] `P0` 20–30B токенов качественного корпуса.
- [ ] `P0` Прогон 1B модели.
- [ ] `P0` Проверка стабильности на всём прогоне.
- [ ] `P0` Оценка на тех же бенчмарках, что и 100M.
- [ ] `P1` Сравнение с 1B Transformer/Mamba baseline.
- [ ] `P1` Оценка стоимости обучения и inference.

### Gate M3

- [ ] 1B обучен без нестабильностей.
- [ ] Качество масштабируется предсказуемо.
- [ ] Есть решение: идти к 8B/80B или менять архитектуру.
- [ ] Есть бюджет и план на следующий масштаб.

---

## 5. M4 — QAT, runtime и 8 GB VRAM

### 5.1 Качество QAT

- [ ] `P0` FP16 baseline и QAT checkpoint на одном чекпоинте.
- [ ] `P0` Замерить падение perplexity/бенчмарков после QAT на 100M.
- [ ] `P0` Подобрать `qat_start_step`, group size и `min_features`.
- [ ] `P0` Проверить QAT на 1B.
- [ ] `P1` Дистилляция/калибровка для восстановления качества.
- [ ] `P1` Сравнение с GPTQ/AWQ/4-bit baseline.
- [ ] `P2` Смешанная точность по слоям.

### 5.2 Runtime

- [x] `P0` Формат AURORA-TQ v1: weights, scales, metadata, checksums.
- [ ] `P0` Быстрый ternary matmul.
- [ ] `P0` C++/CUDA inference runtime.
- [ ] `P0` Memory-mapped загрузка экспертов.
- [ ] `P0` Expert cache/prefetch.
- [ ] `P0` Замер: 8 GB VRAM, 32 GB RAM, целевой tokens/s.
- [ ] `P0` Профилирование узких мест runtime.
- [ ] `P1` Батчинг, streaming и HTTP API на уровне runtime
  (PyTorch-уровень готов).
- [ ] `P1` Fallback на CPU.
- [ ] `P2` Speculative decoding.

### Gate M4

- [ ] QAT-модель достигает согласованного quality target.
- [ ] Runtime укладывается в 8 GB VRAM на целевой конфигурации.
- [ ] Есть воспроизводимый benchmark latency/throughput.
- [ ] Формат весов стабилен и задокументирован (AURORA-TQ v1 готов).

---

## 6. M5 — Масштаб 80B

### 6.1 Инфраструктура

- [ ] `P0` Multi-node training.
- [ ] `P0` Expert parallelism на 140 экспертов.
- [ ] `P0` Sharded checkpoints и быстрое восстановление.
- [ ] `P0` Отказоустойчивость на длинных прогонах.
- [ ] `P0` Оценка стоимости обучения и inference.
- [ ] `P1` Pipeline parallelism при необходимости.
- [ ] `P1` Оптимизация communication overhead.

### 6.2 Данные

- [ ] `P0` 200–400B+ токенов, согласованных с active params.
- [ ] `P0` Финальный data mixture.
- [ ] `P0` Контроль contamination.
- [ ] `P1` Curriculum и long-context stage.

### 6.3 Обучение и оценка

- [ ] `P0` 80B pretraining run.
- [ ] `P0` Промежуточные evals и early diagnostics.
- [ ] `P0` QAT stage.
- [ ] `P0` Финальные бенчмарки.
- [ ] `P0` Сравнение с моделями сопоставимого active compute.
- [ ] `P1` Scaling report.

### Gate M5

- [ ] 80B обучен и воспроизводим.
- [ ] Качество не хуже baseline при равном active compute.
- [ ] Inference укладывается в целевое железо.
- [ ] Есть публичный технический отчёт.

---

## 7. M6 — Alignment, safety и релиз

### 7.1 Post-training

- [ ] `P0` Instruction/SFT dataset.
- [ ] `P0` SFT.
- [ ] `P0` Preference optimization (DPO/ORPO).
- [ ] `P0` Оценка instruction-following.
- [ ] `P1` Tool use / function calling.
- [ ] `P1` Reasoning-данные.
- [ ] `P1` Russian/English alignment.

### 7.2 Safety

- [ ] `P0` Red teaming.
- [ ] `P0` Отказ от опасных запросов.
- [ ] `P0` PII и privacy-проверки.
- [ ] `P0` Bias/toxicity evaluation.
- [ ] `P1` Safety-политика и ограничения использования.

### 7.3 Релиз

- [ ] `P0` Лицензия кода и весов.
- [ ] `P0` Финальная model card обученного 100M/1B (генератор есть).
- [ ] `P0` Финальная data card реального корпуса (генератор есть).
- [ ] `P0` Evaluation report с методологией и mean/std.
- [ ] `P0` Воспроизводимые команды обучения/инференса.
- [ ] `P0` Контрольные суммы весов.
- [ ] `P1` Release notes.
- [ ] `P1` Примеры использования и API docs.

### Gate M6

- [ ] Все `P0` закрыты.
- [ ] Бенчмарки и ограничения опубликованы честно.
- [ ] Нет нерешённых критических safety/legal-проблем.
- [ ] Релиз воспроизводим и задокументирован.

---

## 8. Сквозные требования

### 8.1 Воспроизводимость

- [x] `P0` Git commit, конфиг, seed и версия данных в run metadata.
- [x] `P0` Логи и метрики сохраняются вместе с чекпоинтом.
- [ ] `P0` Эмпирически подтвердить одинаковые результаты при повторном запуске
  в пределах допуска (два seed на 100M).
- [x] `P1` Реестр экспериментов и артефактов (`aurora-registry`).
- [ ] `P1` Гайд по воспроизведению экспериментов.

### 8.2 Производительность

- [x] `P0` Профилирование core/memory/MoE по отдельности (CPU).
- [x] `P0` Замер tokens/s и peak memory (CPU и CUDA).
- [ ] `P0` Устранение лишних GPU-синхронизаций.
- [ ] `P1` Fused kernels для recurrent core.
- [ ] `P1` Chunkwise parallel scan.
- [ ] `P1` CUDA graphs и `torch.compile` без рекомпиляций.
- [ ] `P1` Убрать рекомпиляции MoE при `torch.compile`.

### 8.3 Надёжность

- [x] `P0` Авто-resume после падения.
- [x] `P0` Проверка чекпоинтов на битые тензоры.
- [x] `P0` Мониторинг non-finite loss/grad.
- [x] `P1` Алерты при деградации и раннем останове.
- [x] `P1` Backup чекпоинтов с ротацией.
- [ ] `P1` Внешний мониторинг и дашборды длинных прогонов.

### 8.4 Безопасность и право

- [ ] `P0` Лицензии датасетов проверены.
- [ ] `P0` Лицензия кода и весов определена.
- [ ] `P0` Персональные данные удалены.
- [ ] `P0` Нет секретов в репозитории.
- [ ] `P1` Политика ответственного использования.

### 8.5 Документация

- [x] `P0` README с быстрым стартом.
- [x] `P0` ROADMAP с этапами и чек-листами.
- [x] `P0` Architecture decision records.
- [x] `P0` Описание чекпоинтов и AURORA-TQ.
- [x] `P1` Генератор model card и черновик AURORA-100M.
- [x] `P1` Генератор data card.
- [ ] `P1` API reference.
- [ ] `P1` Troubleshooting.
- [ ] `P1` Гайд по воспроизведению экспериментов.
- [ ] `P2` Схемы обучения и диаграммы.
- [ ] `P2` Технический отчёт/статья.

---

## 9. Ближайший backlog

### 9.1 Можно сделать без реального корпуса

1. [ ] `P0` Определить лицензию и правила использования.
2. [x] `P0` Реализовать hyperparameter sweep runner с генерацией конфигов,
   запуском и сводкой результатов (`aurora-sweep`).
3. [x] `P0` Добавить конфиги 10M/30M/300M и инструмент fit scaling law
   (`aurora-scaling`; эмпирический fit ещё не выполнен).
4. [ ] `P1` Реализовать state reset на границах документов и проверить на smoke.
5. [ ] `P1` Расширить `aurora-ablate` на `n_levels`, `d_state`, `memory_heads`,
   `n_experts`, `top_k`.
6. [ ] `P1` Сделать memory-mapped загрузку экспертов и прототип expert cache.
7. [ ] `P1` Реализовать быстрый ternary matmul и сравнить с dense path.
8. [ ] `P1` Добавить API reference, troubleshooting и reproduction guide.
9. [ ] `P2` Смешанная точность по слоям.
10. [ ] `P2` Схемы обучения и диаграммы.

### 9.2 Требует данных и/или compute

1. [ ] `P0` Собрать и очистить первый реальный RU/EN/code-корпус.
2. [ ] `P0` Обучить токенизатор 64k и замерить fertility.
3. [ ] `P0` Подготовить версионированные train/val/test и data card.
4. [ ] `P0` Прогнать hyperparameter sweep на малом масштабе.
5. [ ] `P0` Запустить AURORA-100M на 1–2B токенов в двух seed.
6. [ ] `P0` Обучить Transformer-100M на тех же данных и compute.
7. [ ] `P0` Прогнать zero-shot/RU/code-бенчмарки и contamination-check.
8. [ ] `P0` Собрать отчёт с mean/std и сделать вывод по гипотезе.
9. [ ] `P1` Написать финальные data card и model card для 100M.
10. [ ] `P1` Принять решение о переходе к абляциям/scaling или смене архитектуры.

---

## 10. Definition of Done для любого эксперимента

- [ ] Зафиксирован git commit.
- [ ] Зафиксированы конфиг, seed и параметры запуска.
- [ ] Зафиксированы версия данных и токенизатора.
- [ ] Сохранены логи, метрики и чекпоинт.
- [ ] Указаны compute budget и железо.
- [ ] Есть baseline для сравнения.
- [ ] Есть вывод и следующий шаг.
- [ ] Результат воспроизводим по инструкции.

---

## 11. Анти-цели

- [ ] Не сравнивать AURORA-80 с DeepSeek-V3/R1 напрямую без равных данных,
  compute и бенчмарков.
- [ ] Не заявлять превосходство по параметрам: 80B total / 11.5B active ≠
  671B total / 37B active.
- [ ] Не запускать 80B, пока не подтверждены scaling laws на 100M–1B.
- [ ] Не считать smoke-обучение доказательством качества архитектуры.
- [ ] Не публиковать QAT-результаты без FP baseline.
- [ ] Не использовать benchmark-данные в training set.
- [ ] Не заявлять 8 GB VRAM без реального runtime-бенчмарка.
- [ ] Не выпускать модель без safety- и legal-проверок.

---

## 12. Критерий успеха проекта

Проект считается успешным, если при равном active compute и равном числе
обученных токенов AURORA показывает конкурентное или лучшее качество на
согласованном наборе бенчмарков, а inference укладывается в целевые 8 GB VRAM.

Если гипотеза не подтверждается, это фиксируется как отрицательный результат с
воспроизводимыми экспериментами, конфигами, логами и чекпоинтами.
