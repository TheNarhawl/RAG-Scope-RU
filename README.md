# RAG-Scope-RU

Локальная песочница для экспериментов с RAG на русскоязычных документах. Меняю по одной переменной — чанкинг, embedding-модель, векторную базу, retrieval-стратегию, LLM — и измеряю, как это влияет на качество ответов.

## Быстрый старт

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
# HF_TOKEN и настройки Qdrant

# PDF кладутся в data/docs/, затем:
python src/ingest.py
python src/retrieve.py "вопрос"
python src/evaluate_retrieval.py --tag baseline-minilm
```

## Стек

**Реализовано:**

- **Embeddings:** sentence-transformers (MiniLM, multilingual-e5-large)
- **Vector store:** Qdrant (Docker)
- **PDF:** pypdf
- **Config:** YAML + .env
- **Метрики:** Hit Rate@1/3/5/10, MRR, ms/query
- **Развёртывание:** Docker Compose + GPU

**Планируется:**

- Reranker, hybrid search
- LLM через Ollama
- RAGAS: Faithfulness, Answer Correctness
- Streamlit

## Датасет

Эксперименты идут на **RRNCB** — Russian RAG Normative Corporate Benchmark.

65 PDF (кодексы, ГОСТы, СНиПы) + 200 вопросов с эталонными ответами и указанием файла-источника.

- Hugging Face: [FractalGPT/RRNCBPublic](https://huggingface.co/datasets/FractalGPT/RRNCBPublic)
- PDF-корпус — по ссылке в README датасета, кладётся в `data/docs/`
- Вопросы конвертируются в `data/eval/questions.json` скриптом `src/prepare_questions.py`

## Структура проекта

```text
RAG-Scope-RU/
├── configs/                    # YAML-конфиги экспериментов
├── data/
│   ├── docs/                   # исходные PDF (не в git)
│   └── eval/                   # questions.json и CSV из RRNCB
├── src/
│   ├── ingest.py               # PDF → чанки → embeddings → Qdrant
│   ├── retrieve.py             # запрос → top-k чанков
│   ├── evaluate_retrieval.py   # метрики на questions.json
│   ├── summarize_experiments.py
│   └── prepare_questions.py
├── experiments/                # результаты прогонов
├── notebooks/                  # анализ и графики
└── docker/                     # конфиги контейнеров
```

## Результаты

Сводка по всем прогонам: [experiments/RESULTS.md](experiments/RESULTS.md)

### Последняя сводка таблицы результатов

_Сгенерировано: 2026-10-17 07:10:40_

Всего прогонов: **5**

| # | Tag | Embedding | Chunk | Overlap | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Faithful | AnsCorr | ms/q | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 200 | 0.130 | 0.225 | 0.305 | — | 0.188 | — | — | — | 20261017_000037 |
| 2 | e5_check | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 48.9 | 20261017_020038 |
| 3 | e5-dense | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 46.4 | 20261017_065800 |
| 4 | e5-hybrid-rerank | multilingual-e5-large | 512 | 64 | 200 | 0.655 | 0.840 | 0.895 | 0.935 | 0.755 | — | — | 290.4 | 20261017_070054 |
| 5 | e5-hybrid | multilingual-e5-large | 512 | 64 | 200 | 0.805 | 0.900 | 0.920 | 0.945 | 0.854 | — | — | 50.0 | 20261017_071007 |

Faithful и AnsCorr (RAGAS) появятся после подключения оценки генерации через LLM + RAGAS.

### Лучший запуск

- **Tag:** `e5_check`
- **Embedding:** `intfloat/multilingual-e5-large`
- **Chunk:** 512 / overlap 64
- **MRR:** 0.825
- **Hit@1:** 0.775
- **Файл:** `metrics_e5_check_20261017_020038.json`

## Запуск через Docker

Для локального запуска и для переноса на удалённый сервер с GPU.

### Требования

- Docker Desktop (Windows/macOS) или Docker Engine + Docker Compose (Linux)
- Для GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Локально

```bash
cd docker

# 1. Собрать образ приложения
docker compose build app

# 2. Поднять Qdrant
docker compose up -d qdrant

# 3. Проиндексировать корпус
docker compose run --rm app python src/ingest.py --config baseline

# 4. Проверить поиск
docker compose run --rm app python src/retrieve.py --config baseline "срок исковой давности"

# 5. Оценить метрики
docker compose run --rm app python src/evaluate_retrieval.py --config baseline --tag docker-minilm

# 6. Остановить Qdrant
docker compose down
```


## Лицензия

TBD.