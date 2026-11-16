# AURORA-80 — полный план этапов

> Цель: довести исследовательскую гипотезу AURORA до проверяемого результата, не пропустив ни один критический этап.
>
> Легенда:
> - `[x]` — сделано и проверено;
> - `[ ]` — не сделано;
> - `P0` — блокирует переход к следующему этапу;
> - `P1` — важно для качества/масштабирования;
> - `P2` — улучшение/полировка.
>
> Правило: этап закрывается только когда отмечены все `P0` и пройден gate этапа. Нельзя заявлять превосходство над другой моделью без равных данных, compute и единого протокола оценки.

---

## 0. Текущий срез проекта

### 0.1 Что уже есть

- [x] Репозиторий, Python-пакет `aurora`, `pyproject.toml`, `.gitignore`.
- [x] Русскоязычный `README.md` с архитектурой, математикой и инструкциями.
- [x] Git-репозиторий, SSH remote, первый коммит отправлен в `origin/main`.
- [x] Конфиги `smoke`, `100m`, `100m_qat`, `transformer_100m`, `1b`, `80b`.
- [x] Расчёт параметров: 100M — 105.73M total / 49.10M active; 1B — 929.49M / 325.51M; 80B — 80.49B / 11.52B; Transformer-100M baseline — 106.28M.
- [x] 168 тестов; `ruff`, `mypy` проходят.
- [x] Smoke-обучение, resume, генерация, perplexity, QAT-обучение и QAT-inference.
- [x] Демонстрационный корпус и скрипт подсчёта параметров.
- [x] CI, `Makefile`, потоковая подготовка данных с manifest.
- [x] Провенанс экспериментов: run metadata в checkpoint, JSONL-метрики и eval report.
- [x] `L_memory`/`L_consistency`, throughput/RAM benchmark, Transformer baseline, eval harness.
- [x] Потоковое курирование корпуса: exact/near dedup, фильтры качества, PII drop/redact и JSON-отчёт.
- [x] Потоковая проверка benchmark contamination по word n-граммам (`aurora-contamination`).
- [x] Потоковая статистика корпуса: домены, письменности, code-строки и `language_hint` (`aurora-corpus-stats`).
- [x] Zero-shot multiple-choice harness с log-prob скорингом (`aurora-zero-shot`; прогон на реальных бенчмарках — после 100M).
- [x] Опциональный W&B-трекер (online/offline) с graceful fallback.
- [x] Offline-калибровка групповых ternary-масштабов (`--calibrate`).
- [x] Сравнение FP32/FP16/ternary/INT8/INT4 на одном чекпоинте и датасете (`aurora-quant-bench`).
- [x] Проверка стабильности состояния на 4k/8k/16k/32k с опциональным reset (`aurora-state-check`; smoke/CPU).
- [x] Абляция `core`/`core_memory`/`full` с равным token budget (`aurora-ablate`; smoke/CPU).
- [x] CUDA AMP (BF16, FP16+GradScaler), gradient checkpointing и `torch.compile` (`aot_eager`, `inductor`) на RTX 3050; замеры 100M (CUDA медленнее CPU; MoE recompiles остаются).

### 0.2 Чего принципиально ещё нет

- [ ] Реального обучения на большом корпусе.
- [ ] Публичных бенчмарков и сравнения с baseline-архитектурами.
- [ ] Обученного токенизатора на 64k/128k.
- [ ] Distributed training, FSDP/TP/EP.
- [ ] Chunkwise parallel scan и быстрых ядер.
- [ ] C++/CUDA runtime и заявленных 8 GB VRAM.
- [ ] SFT, alignment, safety-оценки.
- [ ] Финальные лицензии, model card и data card для обученного 100M (генераторы и черновики готовы).
- [ ] Доказательства превосходства над DeepSeek или любой другой моделью.

---

## 1. M0 — Foundation: воспроизводимый прототип

### 1.1 Инженерия и инфраструктура

- [x] `P0` Структура пакета и entry points (`aurora-train`, `aurora-prepare`, `aurora-generate`, `aurora-perplexity`, `aurora-evaluate`, `aurora-tokenizer`).
- [x] `P0` Строгая валидация YAML-конфигов через dataclass.
- [x] `P0` Seed, deterministic-режим, сохранение/восстановление RNG.
- [x] `P0` Checkpoint save/load, atomic replace, `weights_only=True`.
- [x] `P0` Resume обучения, включая QAT.
- [x] `P0` Провенанс экспериментов: run metadata (git, окружение, dataset fingerprint) в checkpoint, JSONL-метрики и eval report.
- [x] `P0` Проверка `manifest.dataset_version`, размеров и SHA-256 для train/val/mixture при запуске и resume.
- [x] `P1` Опциональный TensorBoard.
- [x] `P1` Опциональный `torch.compile` (код есть).
- [x] `P0` Проверить `torch.compile` на 100M конфигурации (`aot_eager` и `inductor` на CUDA; для `inductor` понадобились локальные заголовки Python 3.12; MoE recompiles остаются).
- [x] `P0` CI: lint, mypy, tests и end-to-end smoke на каждый push.
- [x] `P1` Docker-образ для обучения и inference (CPU по умолчанию, CUDA через build arg; сборка проверяется в CI).
- [x] `P1` Makefile/Taskfile с типовыми командами.
- [ ] `P2` Публикация пакета/версионирование.

### 1.2 Архитектура

- [x] `P0` Многоуровневое рекуррентное ядро с межуровневым взаимодействием.
- [x] `P0` Ассоциативная память с delta rule, decay/write gates.
- [x] `P0` MoE с Top-K, load-balancing loss и z-loss.
- [x] `P0` Residual integration и LayerScale.
- [x] `P0` RMSNorm, SwiGLU, tied embeddings.
- [x] `P0` `LanguageHead` и корректная инициализация.
- [x] `P0` Причинность и продолжение состояния проверены тестами.
- [x] `P1` `norm_eps` прокинут во внутренние нормы.
- [x] `P1` Реализовать `L_memory` и `L_consistency`.
- [x] `P1` Провести ablation: вклад core/memory/MoE по отдельности (smoke/CPU; содержательный вывод — на 100M в M2).
- [x] `P1` Проверить стабильность состояния на длинах 4k/8k/16k/32k (smoke-модель, CPU; 100M — после M1).
- [x] `P1` Решить, нужны ли начальное состояние и state reset между документами (ADR 0004: нулевое начальное состояние; reset на границах документов — в M2).
- [ ] `P2` Варианты памяти: linear attention, delta-net, Titans-like.
- [ ] `P2` Shared experts и expert choice routing.

### 1.3 Обучение

- [x] `P0` AdamW с раздельным weight decay.
- [x] `P0` Cosine scheduler с warmup.
- [x] `P0` BF16/FP16 autocast и GradScaler для FP16.
- [x] `P0` Gradient accumulation, clipping, пропуск non-finite шагов.
- [x] `P0` Validation loss/perplexity и best/latest checkpoints.
- [x] `P0` Smoke-обучение подтверждает снижение loss.
- [x] `P0` Проверить AMP на реальной CUDA (RTX 3050; BF16 и FP16+GradScaler; smoke train/resume).
- [x] `P0` Замерить throughput и peak memory для 100M (CPU и CUDA: 22 tok/s, 2.7 GB allocated, 3.2 GB reserved на CUDA).
- [x] `P1` Gradient checkpointing для длинных последовательностей (CPU-тесты; CUDA: −24% VRAM, −41% throughput).
- [ ] `P1` Подобрать learning rate, betas, warmup, weight decay на 100M.
- [x] `P1` Early stopping и остановка по compute budget (`max_tokens`).
- [x] `P1` Эксперимент-трекер: опциональный W&B с online/offline режимами и graceful fallback.
- [ ] `P2` Curriculum learning и context extension.

### 1.4 Данные и токенизация

- [x] `P0` Byte-level tokenizer.
- [x] `P0` SentencePiece tokenizer и скрипт обучения.
- [x] `P0` Подготовка `.bin` и memmap-датасет.
- [x] `P0` Защита от утечки validation в train.
- [x] `P0` Проверка границ `uint16`/`uint32`.
- [x] `P0` Pickle-safe memmap для DataLoader workers.
- [ ] `P0` Собрать русский, английский и code-корпус.
- [ ] `P0` Обучить токенизатор 64k/128k на реальном корпусе.
- [ ] `P0` Оценить fertility/compression токенизатора для RU/EN/code (инструмент `aurora-tokenizer-stats` готов; нужен 64k-токенизатор и реальный корпус).
- [x] `P0` Инструмент дедупликации, фильтрации качества и удаления PII (`aurora-curate`; прогон на реальном корпусе — M1).
- [x] `P1` Потоковый `prepare.py` без загрузки всего корпуса в RAM.
- [x] `P1` Data mixture с весами доменов (`train_paths`/`train_weights`).
- [x] `P1` Train/val/test split без пересечения документов (`--split-mode document`, `--test-fraction`).
- [x] `P1` Data card и версионирование датасета (`dataset_version`, `aurora-data-card`, `--verify`).
- [ ] `P2` Синтетические данные и code/math-специализация.

### 1.5 QAT и сжатие

- [x] `P0` Ternary quantization, STE, packing/unpacking.
- [x] `P0` `TernaryLinear`, конвертация модели, идемпотентность.
- [x] `P0` QAT checkpoint loading для генерации и perplexity.
- [x] `P0` Тесты QAT-обучения и QAT-inference.
- [ ] `P0` Измерить quality recovery после QAT на 100M.
- [ ] `P0` Реализовать быстрый ternary matmul/kernel.
- [x] `P1` Формат сериализации AURORA-TQ (`save_tq`/`load_tq`, CLI `aurora-tq`).
- [x] `P1` Сравнить ternary с 4-bit/8-bit и FP16 baseline (weight-only PTQ harness; прогон на 100M — после M1-обучения).
- [x] `P1` Калибровка групповых масштабов (offline по ошибке реконструкции весов; активационная калибровка не реализована).
- [ ] `P2` Смешанная точность по слоям.

### 1.6 Inference

- [x] `P0` PyTorch reference inference.
- [x] `P0` State continuation без KV-cache.
- [x] `P0` Temperature, top-k, top-p, repetition penalty.
- [x] `P0` Centralized checkpoint loading и autocast.
- [x] `P0` Perplexity evaluation.
- [x] `P0` Замерить latency/tokens/s/RAM для 100M (CPU и CUDA; CUDA eager медленнее из-за последовательного ядра).
- [x] `P0` Реализовать батчевый и streaming inference (`generate_batch`, `stream_generate`).
- [x] `P1` HTTP API/server (`aurora-serve`; single-prompt, localhost by default).
- [ ] `P1` Memory-mapped загрузка экспертов.
- [ ] `P1` Expert prefetch/cache для 8 GB VRAM.
- [ ] `P2` Speculative decoding.

### 1.7 Качество и документация

- [x] `P0` 168 тестов на архитектуру, память, роутер, QAT, обучение, данные, курирование, статистику корпуса, contamination, чекпоинты, провенанс, gradient checkpointing, mixture, document split, HTTP API, AURORA-TQ, data card, model card, zero-shot оценку, W&B tracker, сравнение квантизации, длинный контекст, абляцию компонентов и inference.
- [x] `P0` `ruff`, `mypy`, инструкции запуска.
- [x] `P0` README с ограничениями и научной базой.
- [x] `P0` Зафиксировать baseline-метрики smoke-модели.
- [x] `P1` Architecture decision records (ADR).
- [x] `P1` Генератор model card (`aurora-model-card`) и черновик `docs/model_card_100m.md`.
- [ ] `P2` Схемы обучения и диаграммы.

### Gate M0

- [x] Все тесты проходят.
- [x] Smoke-обучение воспроизводимо.
- [x] Checkpoint можно сохранить, загрузить и использовать.
- [x] QAT-модель можно обучить и запустить.
- [x] CI зелёный на чистой машине (lint, mypy, pytest и end-to-end smoke с resume и проверкой провенанса).
- [x] Замерены throughput, память и latency прототипа (CPU и CUDA).

---

## 2. M1 — Research baseline 100M

**Цель:** получить воспроизводимый baseline на реальном корпусе и сравнить с Transformer-аналогом при равном compute.

### 2.1 Данные

- [ ] `P0` Корпус не менее 2–5 млрд токенов после токенизации.
- [ ] `P0` Зафиксированная версия датасета и хэши файлов.
- [ ] `P0` Очистка, дедупликация, фильтр качества.
- [ ] `P0` Отдельный test set, не использованный в обучении.
- [ ] `P0` Обученный токенизатор 64k.
- [x] `P1` Анализ распределения языков и доменов (`aurora-corpus-stats`; эвристика по письменностям; прогон на реальном корпусе — после его сборки).
- [x] `P1` Защита от benchmark contamination (word n-gram checker `aurora-contamination`; прогон на реальных бенчмарках — после сборки корпуса).

### 2.2 Обучение

- [x] `P1` Реализовать Transformer baseline того же размера.
- [ ] `P0` Полный прогон 100M на 1–2B токенов.
- [ ] `P0` Несколько seed-ов или минимум 2 повторных запуска.
- [ ] `P0` Логи loss/LR/grad norm/tokens/s.
- [ ] `P0` Checkpoint каждые N шагов, resume проверен.
- [ ] `P1` Hyperparameter sweep.
- [ ] `P1` Сравнение с Transformer baseline того же размера.
- [ ] `P1` Сравнение с Mamba-2/RWKV-7 baseline, если доступны.
- [ ] `P2` Ablation по `n_levels`, `d_state`, `memory_heads`, `n_experts`, `top_k`.

### 2.3 Оценка

- [x] `P1` Единый eval harness с per-domain perplexity и JSON-отчётом.
- [ ] `P0` Held-out perplexity на RU/EN/code.
- [ ] `P0` Zero-shot: MMLU, HellaSwag, ARC, WinoGrande (harness `aurora-zero-shot` готов; нужны данные и обученный чекпоинт).
- [ ] `P0` RU-бенчмарки: ruMMLU, Russian SuperGLUE и аналоги.
- [ ] `P0` Code: HumanEval, MBPP.
- [ ] `P0` Отчёт с таблицей результатов и погрешностями.
- [ ] `P1` Анализ ошибок и типичных failure modes.
- [ ] `P1` Human evaluation для RU/EN.

### Gate M1

- [ ] Loss стабильно снижается, нет деградации состояния.
- [ ] Модель превосходит random baseline.
- [ ] Есть честное сравнение с Transformer при равных параметрах/токенах.
- [ ] Есть вывод: гипотеза подтверждается, опровергается или требует изменений.
- [ ] Все эксперименты воспроизводятся по README.

---

## 3. M2 — Архитектурные абляции и scaling laws

### 3.1 Абляции

- [ ] `P0` Core only.
- [ ] `P0` Core + memory.
- [ ] `P0` Core + memory + MoE.
- [ ] `P0` Варианты residual integration и LayerScale.
- [ ] `P0` Вклад `L_memory` и `L_consistency`.
- [ ] `P0` Влияние `n_levels` и `d_state`.
- [ ] `P0` Влияние числа экспертов и `top_k`.
- [ ] `P1` Варианты decay/write gates.
- [ ] `P1` Варианты нормализации состояния.
- [ ] `P1` Shared experts vs independent experts.

### 3.2 Scaling laws

- [ ] `P0` Серия 10M/30M/100M/300M параметров.
- [ ] `P0` Fit scaling law по loss от compute/parameters/tokens.
- [ ] `P0` Проверка предсказания на 300M.
- [ ] `P1` Оценка compute-optimal token budget.
- [ ] `P1` Сравнение scaling curve с Transformer baseline.

### 3.3 Длинный контекст

- [ ] `P0` Обучение/оценка на 2k/4k/8k.
- [ ] `P0` Проверка деградации памяти и состояния.
- [ ] `P1` Chunkwise parallel scan.
- [ ] `P1` State reset и документные границы.
- [ ] `P1` Оценка long-context задач.
- [ ] `P2` Расширение до 32k+.

### Gate M2

- [ ] Выбрана финальная архитектура для 1B.
- [ ] Есть scaling law с приемлемой ошибкой.
- [ ] Понятно, какие компоненты дают вклад, а какие можно удалить.

---

## 4. M3 — Масштаб 1B

### 4.1 Инфраструктура

- [ ] `P0` Distributed Data Parallel.
- [ ] `P0` FSDP или sharded optimizer.
- [ ] `P0` Sharded checkpoints.
- [ ] `P0` Expert parallelism для MoE.
- [ ] `P0` Мониторинг GPU/RAM/network.
- [ ] `P1` Mixed precision на реальном кластере.
- [ ] `P1` Gradient/activation checkpointing.
- [ ] `P1` Отказоустойчивость и авто-resume.

### 4.2 Данные и обучение

- [ ] `P0` 20–30B токенов качественного корпуса.
- [ ] `P0` Прогон 1B модели.
- [ ] `P0` Проверка стабильности на всём прогоне.
- [ ] `P0` Оценка на тех же бенчмарках, что и 100M.
- [ ] `P1` Сравнение с 1B Transformer/Mamba baseline.
- [ ] `P1` Оценка стоимости обучения.

### Gate M3

- [ ] 1B обучен без нестабильностей.
- [ ] Качество масштабируется предсказуемо.
- [ ] Есть решение: идти к 8B/80B или менять архитектуру.
- [ ] Есть бюджет и план на следующий масштаб.

---

## 5. M4 — QAT, runtime и 8 GB VRAM

### 5.1 Качество QAT

- [ ] `P0` FP16 baseline и QAT checkpoint на одном чекпоинте.
- [ ] `P0` Замер падения perplexity/бенчмарков после QAT.
- [ ] `P0` Подбор `qat_start_step`, group size, min features.
- [ ] `P0` Проверка QAT на 1B.
- [ ] `P1` Дистилляция/калибровка для восстановления качества.
- [ ] `P1` Сравнение с GPTQ/AWQ/4-bit baseline.

### 5.2 Runtime

- [x] `P0` Формат AURORA-TQ: weights, scales, metadata, versioning (v1, checksums).
- [ ] `P0` Быстрый ternary matmul.
- [ ] `P0` C++/CUDA inference runtime.
- [ ] `P0` Memory-mapped эксперты.
- [ ] `P0` Expert cache/prefetch.
- [ ] `P0` Замер: 8 GB VRAM, 32 GB RAM, целевой tokens/s.
- [ ] `P1` Батчинг, streaming, HTTP API (PyTorch-уровень готов; оптимизированный runtime — нет).
- [ ] `P1` Профилирование узких мест.
- [ ] `P1` Fallback на CPU.

### Gate M4

- [ ] QAT-модель достигает согласованного quality target.
- [ ] Runtime укладывается в 8 GB VRAM на целевой конфигурации.
- [ ] Есть воспроизводимый benchmark latency/throughput.
- [x] Формат весов задокументирован и стабилен (AURORA-TQ v1).

---

## 6. M5 — Масштаб 80B

### 6.1 Инфраструктура

- [ ] `P0` Multi-node training.
- [ ] `P0` Expert parallelism на 140 экспертов.
- [ ] `P0` Sharded checkpoints и быстрое восстановление.
- [ ] `P0` Отказоустойчивость на длинных прогонах.
- [ ] `P0` Оценка стоимости обучения и инференса.
- [ ] `P1` Pipeline parallelism при необходимости.
- [ ] `P1` Оптимизация communication overhead.

### 6.2 Данные

- [ ] `P0` 200–400B+ токенов (согласовано с active params).
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

- [ ] `P0` Лицензия.
- [ ] `P0` Финальная model card обученного 100M (генератор есть).
- [ ] `P0` Финальная data card реального корпуса (генератор есть).
- [ ] `P0` Evaluation report с методологией.
- [ ] `P0` Воспроизводимые команды обучения/инференса.
- [ ] `P0` Контрольные суммы весов.
- [ ] `P1` Release notes.
- [ ] `P1` Примеры использования и API docs.

### Gate M6

- [ ] Все P0 закрыты.
- [ ] Бенчмарки и ограничения опубликованы честно.
- [ ] Нет нерешённых критических safety/legal-проблем.
- [ ] Релиз воспроизводим и задокументирован.

---

## 8. Сквозные требования

### 8.1 Воспроизводимость

- [ ] `P0` Каждый эксперимент имеет git commit, конфиг, seed и версию данных.
- [ ] `P0` Логи и метрики сохраняются вместе с чекпоинтом.
- [ ] `P0` Одинаковые результаты при повторном запуске в пределах допуска.
- [ ] `P1` Реестр экспериментов и артефактов.

### 8.2 Производительность

- [x] `P0` Профилирование core/memory/MoE по отдельности (CPU).
- [x] `P0` Замер tokens/s и peak memory (CPU и CUDA).
- [ ] `P0` Устранение лишних GPU-синхронизаций.
- [ ] `P1` Fused kernels для recurrent core.
- [ ] `P1` Chunkwise parallel scan.
- [ ] `P1` CUDA graphs и `torch.compile` без рекомпиляций (`inductor` проверен; CUDA graphs — нет).
- [ ] `P1` Убрать рекомпиляции MoE при `torch.compile` (`inductor` на CUDA прогнан; достигается recompile limit).

### 8.3 Надёжность

- [x] `P0` Авто-resume после падения.
- [x] `P0` Проверка чекпоинтов на битые тензоры.
- [x] `P0` Мониторинг non-finite loss/grad.
- [ ] `P1` Алерты при деградации.
- [ ] `P1` Backup чекпоинтов.

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
- [x] `P0` Описание формата чекпоинтов и AURORA-TQ.
- [x] `P1` Генератор model card и черновик карты AURORA-100M.
- [ ] `P1` API reference.
- [ ] `P1` Troubleshooting.
- [ ] `P1` Гайд по воспроизведению экспериментов.
- [ ] `P2` Технический отчёт/статья.

---

## 9. Ближайший P0-бэклог

1. [ ] Определить лицензию и правила использования.
2. [ ] Собрать и очистить первый реальный корпус.
3. [ ] Обучить токенизатор 64k и замерить fertility.
4. [x] Сделать потоковую подготовку данных.
5. [ ] Запустить 100M на 1–2B токенов.
6. [x] Реализовать Transformer baseline того же размера.
7. [x] Добавить единый eval harness и отчёт (нужен прогон на 100M).
8. [x] Замерить throughput/VRAM/latency (CPU и CUDA; 100M ~2.7 GB VRAM).
9. [x] Проверить AMP и `torch.compile` на CUDA (BF16, FP16+GradScaler, `aot_eager`, `inductor`; MoE recompiles остаются).
10. [x] Реализовать `L_memory` и `L_consistency` (smoke-абляция есть; вклад на 100M ещё нужен).
11. [x] Добавить CI.
12. [x] Реализовать потоковое курирование корпуса: dedup, quality filters, PII.
13. [ ] Написать финальные data card и model card для обученного 100M (генераторы готовы; нужны реальные данные и чекпоинт).
14. [x] Реализовать генератор model card (`aurora-model-card`) и черновик `docs/model_card_100m.md`.
15. [x] Реализовать n-gram проверку benchmark contamination (`aurora-contamination`).
16. [x] Добавить опциональный W&B experiment tracker.
17. [x] Реализовать zero-shot multiple-choice harness (`aurora-zero-shot`).

---

## 10. Definition of Done для любого эксперимента

- [ ] Зафиксирован git commit.
- [ ] Зафиксирован конфиг и seed.
- [ ] Зафиксирована версия данных и токенизатора.
- [ ] Сохранены логи, метрики и чекпоинт.
- [ ] Указан compute budget и железо.
- [ ] Есть baseline для сравнения.
- [ ] Есть вывод и следующий шаг.
- [ ] Результат воспроизводим по инструкции.

---

## 11. Анти-цели и честные ограничения

- [ ] Не сравнивать AURORA-80 с DeepSeek-V3/R1 напрямую без равных данных, compute и бенчмарков.
- [ ] Не заявлять превосходство по параметрам: 80B total / 11.5B active ≠ 671B total / 37B active.
- [ ] Не запускать 80B, пока не подтверждены scaling laws на 100M–1B.
- [ ] Не считать smoke-обучение доказательством качества архитектуры.
- [ ] Не публиковать QAT-результаты без FP baseline.
- [ ] Не использовать benchmark-данные в training set.
- [ ] Не заявлять 8 GB VRAM без реального runtime-бенчмарка.
- [ ] Не выпускать модель без safety- и legal-проверок.

---

## 12. Критерий успеха проекта

Проект считается успешным, если при **равном active compute и равном числе обученных токенов** AURORA показывает конкурентное или лучшее качество на согласованном наборе бенчмарков, а inference укладывается в целевые 8 GB VRAM. Если гипотеза не подтверждается, это фиксируется как отрицательный результат с воспроизводимыми экспериментами.
