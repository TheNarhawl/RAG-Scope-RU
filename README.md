# RAG-Scope-RU

Модульная экспериментальная платформа (песочница) для исследования и оценки качества этапа поиска (Retrieval) в RAG-системах на русскоязычных нормативно-технических документах.

Платформа позволяет изолированно менять и замерять влияние ключевых параметров пайплайна:
* Модели эмбеддингов (от легковесных до SOTA мультиязычных и адаптированных под русский язык)
* Стратегии поиска: плотный (Dense), разреженный лексический (BM25) и гибридный с Reciprocal Rank Fusion (RRF)
* Переранжирование кандидатов (Cross-Encoder Reranker)
* Параметры нарезки текста (размер чанка и перекрытие)
* Метрики качества извлечения и задержка ответа

---

## Быстрый старт

### 1. Окружение и зависимости

```bash
# Клонирование и переход в проект
git clone <URL_РЕПОЗИТОРИЯ>
cd RAG-Scope-RU

# Создание и активация виртуального окружения
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Заполните QDRANT_URL и HF_TOKEN (при необходимости)
```

### 2. Подготовка данных

Скрипт `src/prepare_data.py` в автоматическом режиме:
1. Выкачивает CSV-датасет RRNCB с Hugging Face и конвертирует его в `data/eval/questions.json`.
2. Скачивает полный корпус из 65 нормативных PDF-документов из Google Drive напрямую в `data/docs/`.

```bash
python src/prepare_data.py
```

### 3. Генерация матрицы конфигураций

Генерирует полную сетку YAML-конфигов под разные модели, размеры чанков и стратегии поиска:

```bash
python make_configs.py
```

### 4. Индексация и запуск экспериментов

Индексация выполняется один раз на пару **«Модель + Размер чанка»**, после чего оцениваются все три стратегии поиска:

```bash
# Индексация документов (Dense + Sparse BM25 в Qdrant)
python src/ingest.py --config e5_large/c512/dense

# Проверка поиска вручную
python src/retrieve.py --config e5_large/c512/hybrid "срок исковой давности"

# Оценка метрик ранжирования (Hit@K, MRR, latency)
python src/evaluate_retrieval.py --config e5_large/c512/dense
python src/evaluate_retrieval.py --config e5_large/c512/hybrid
python src/evaluate_retrieval.py --config e5_large/c512/hybrid-rerank

# Сборка единого отчета по всем запускам в experiments/RESULTS.md
python src/summarize_experiments.py
```

---

## Стек технологий

**Реализовано:**
- **Векторное хранилище:** Qdrant (Docker / Cloud / Local)
- **Dense Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, `multilingual-e5-small/base/large`, `deepvk/USER-bge-m3`, `BAAI/bge-m3`, `ai-forever/ru-en-RoSBERTa`, `cointegrated/rubert-tiny2`)
- **Sparse Embeddings (BM25):** `fastembed` с поддержкой русской морфологии
- **Гибридный поиск:** Qdrant Prefetch + Reciprocal Rank Fusion (RRF)
- **Reranker:** Cross-Encoder (`BAAI/bge-reranker-v2-m3`, `BAAI/bge-reranker-base`)
- **Парсинг документов:** `pypdf`
- **Метрики качества:** Hit Rate@1/3/5/10, MRR, ms/query
- **Инфраструктура:** Docker, Docker Compose (NVIDIA GPU passthrough), PowerShell скрипты обслуживания Qdrant

**В разработке:**
- Генеративный контур (генерация ответа через Ollama / Gemini API)
- Сквозная оценка генерации по методологии RAGAS (Faithfulness, Answer Correctness)
- Мульти-LLM судейство (LLM-as-a-Judge) с расчетом согласованности судей (Fleiss / Cohen's Kappa)
- Интерактивный веб-интерфейс на Streamlit

---

## Датасет

Эксперименты проводятся на бенчмарке **RRNCB** (Russian RAG Normative Corporate Benchmark):
* **Корпус:** 65 нормативно-технических PDF (ГОСТы, СНиПы, федеральные законы, кодексы).
* **Тестовая выборка:** 200 контрольных вопросов с эталонными ответами и ссылкой на исходный документ.
* **Hugging Face:** [FractalGPT/RRNCBPublic](https://huggingface.co/datasets/FractalGPT/RRNCBPublic)

Скрипт `src/prepare_data.py` производит потоковую загрузку корпуса и разметки без необходимости ручного скачивания.

---

## Структура проекта

```text
RAG-Scope-RU/
├── configs/                    # Иерархические YAML-конфиги экспериментов
│   ├── baseline.yaml           # Контрольный англоязычный бейслайн
│   ├── e5_large/               # Модель: intfloat/multilingual-e5-large
│   │   ├── c512/               # Размер чанка: 512 символов
│   │   │   ├── dense.yaml
│   │   │   ├── hybrid.yaml
│   │   │   └── hybrid-rerank.yaml
│   │   └── ...
│   ├── bge_m3/                 # Модель: BAAI/bge-m3
│   ├── user_bge_m3/            # Модель: deepvk/USER-bge-m3
│   └── ...
├── data/
│   ├── docs/                   # Корпус PDF (загружается автоматически, не в git)
│   └── eval/                   # dataset_sample.csv и questions.json
├── docker/
│   ├── Dockerfile              # Образ Python 3.12 + PyTorch CUDA
│   ├── docker-compose.yml      # Сервисы App + Qdrant (GPU)
│   ├── reset-qdrant.ps1        # Полный сброс хранилища Qdrant
│   └── restart-qdrant.ps1      # Мягкий перезапуск контейнера Qdrant
├── experiments/                # Результаты прогонов в JSON и сводный отчет
│   └── RESULTS.md              # Актуальная таблица метрик всех экспериментов
├── notebooks/
│   └── results.ipynb           # Анализ распределения метрик и графики
├── src/
│   ├── paths.py                # Централизованные пути и переменные окружения
│   ├── prepare_data.py         # Выкачка датасета с HF и PDF из GDrive
│   ├── ingest.py               # Чанкинг, расчет эмбеддингов, загрузка в Qdrant
│   ├── retrieve.py             # Инференс поиска (Dense / Hybrid / Rerank)
│   ├── evaluate_retrieval.py   # Пакетная оценка качества поиска на вопросах
│   └── summarize_experiments.py# Построение сводной Markdown-таблицы результатов
├── make_configs.py             # Автогенератор дерева конфигураций
├── requirements.txt            # Зависимости проекта
└── README.md
```

---

## Запуск через Docker

Контейнеризация предназначена как для изолированного запуска локально, так и для бесшовного переноса стенда на арендованные GPU-хосты (RunPod, Vast.ai, Cloud.ru, Selectel).

### Локально / на GPU-сервере

```bash
cd docker

# 1. Поднять сервис Qdrant
docker compose up -d qdrant

# 2. Собрать контейнер приложения
docker compose build app

# 3. Выкачать данные корпуса и вопросов внутри контейнера
docker compose run --rm app python src/prepare_data.py

# 4. Сгенерировать матрицу конфигов
docker compose run --rm app python make_configs.py

# 5. Проиндексировать выбранную связку
docker compose run --rm app python src/ingest.py --config e5_large/c512/dense

# 6. Запустить расчет метрик
docker compose run --rm app python src/evaluate_retrieval.py --config e5_large/c512/hybrid

# 7. Сформировать сводку
docker compose run --rm app python src/summarize_experiments.py

# 8. Остановка окружения
docker compose down
```

---

## Текущие результаты экспериментов

Сводная таблица формируется автоматически скриптом `src/summarize_experiments.py` в файле `experiments/RESULTS.md`.

| # | Tag | Embedding | Chunk | Overlap | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Faithful | AnsCorr | ms/q | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 200 | 0.130 | 0.225 | 0.305 | — | 0.188 | — | — | — | 20261017_000037 |
| 2 | e5_check | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 48.9 | 20261017_020038 |
| 3 | e5-dense | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 46.4 | 20261017_065800 |
| 4 | e5-hybrid-rerank | multilingual-e5-large | 512 | 64 | 200 | 0.655 | 0.840 | 0.895 | 0.935 | 0.755 | — | — | 290.4 | 20261017_070054 |
| 5 | e5-hybrid | multilingual-e5-large | 512 | 64 | 200 | 0.805 | 0.900 | 0.920 | 0.945 | 0.854 | — | — | 50.0 | 20261017_071007 |

*Колонки Faithful и AnsCorr резервируют метрики качества генерации фреймворка RAGAS.*

### Лучший запуск на текущий момент

* **Конфигурация:** `e5-hybrid`
* **Embedding модель:** `intfloat/multilingual-e5-large` + `Qdrant/bm25` (RRF)
* **Размер чанка:** 512 символов / перекрытие 64
* **MRR:** 0.854
* **Hit@1:** 0.805 (Hit@10: 0.945)
* **Среднее время запроса:** 50.0 мс