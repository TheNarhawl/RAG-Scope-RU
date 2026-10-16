# RAG-Scope-RU

Локальная песочница для экспериментов с RAG на русскоязычных документах. 

## Быстрый старт

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env      # вставь HF_TOKEN

# PDF-документы лежат в data/docs/
python src/ingest.py
python src/retrieve.py "     вопрос     "

## Актуальные результаты экспериментов

_Сгенерировано: 2026-10-17 00:14:04_

Всего прогонов: **1**

| # | Tag | Embedding | Chunk | Overlap | N | Hit@1 | Hit@3 | Hit@5 | MRR | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 200 | 0.130 | 0.225 | 0.305 | 0.188 | 20261017_000037 |