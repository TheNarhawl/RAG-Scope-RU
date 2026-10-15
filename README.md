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