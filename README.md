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

- **Embeddings:** `sentence-transformers`
- **Vector store:** Qdrant (Docker)
- **PDF:** `pypdf`
- **Config:** YAML + `.env`
- **Метрики:** Hit Rate@K, MRR

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

_Сгенерировано: 2026-10-17 01:46:03_

Всего прогонов: **2**

| # | Tag | Embedding | Chunk | Overlap | N | Hit@1 | Hit@3 | Hit@5 | MRR | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 200 | 0.130 | 0.225 | 0.305 | 0.188 | 20261017_000037 |
| 2 | qdrant_e5_large | multilingual-e5-large | 512 | 64 | 200 | 0.730 | 0.860 | 0.895 | 0.794 | 20261017_014537 |


## Лицензия

TBD.