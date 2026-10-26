import argparse
import csv
import json
import sys
from pathlib import Path

import gdown
import requests

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
EVAL_DIR = ROOT / "data" / "eval"

HF_DATASET_URL = (
    "https://huggingface.co/datasets/FractalGPT/RRNCBPublic/resolve/main/rrncb_public_dataset.csv"
)
GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1B12Y-QX9UfI9RDJDZ8KZkfF7FUz5q3hy"
)

CSV_PATH = EVAL_DIR / "dataset_sample.csv"
OUT_PATH = EVAL_DIR / "questions.json"


def download_eval_dataset():
    """Скачивает CSV датасета RRNCB с Hugging Face."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"==> Скачивание датасета с Hugging Face...")
    response = requests.get(HF_DATASET_URL, stream=True, timeout=30)
    response.raise_for_status()

    with open(CSV_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"    Датасет сохранен: {CSV_PATH}")


def convert_questions():
    """Конвертирует CSV в формат questions.json для evaluate_retrieval.py."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Файл {CSV_PATH} не найден. Сначала скачайте датасет.")

    questions = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            questions.append({
                "id": f"q{i:03d}",
                "question": row["question"],
                "ground_truth": row["answer"],
                "source_hint": row["document"],
            })

    OUT_PATH.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"    Сконвертировано {len(questions)} вопросов в {OUT_PATH}")


def download_pdfs():
    """Скачивает корпус PDF-файлов из Google Drive папки."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"==> Скачивание корпуса PDF из Google Drive в {DOCS_DIR}...")
    
    # gdown умеет выкачивать публичную папку целиком
    gdown.download_folder(
        url=GDRIVE_FOLDER_URL,
        output=str(DOCS_DIR),
        quiet=False,
        use_cookies=False,
    )
    
    pdf_count = len(list(DOCS_DIR.glob("*.pdf")))
    print(f"    Готово! Найдено PDF-файлов в папке: {pdf_count}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для RAG-Scope-RU")
    parser.add_argument("--skip-pdfs", action="store_true", help="Пропустить скачивание PDF")
    parser.add_argument("--skip-questions", action="store_true", help="Пропустить скачивание и конвертацию вопросов")
    args = parser.parse_args()

    if not args.skip_questions:
        download_eval_dataset()
        convert_questions()

    if not args.skip_pdfs:
        download_pdfs()

    print("\nПодготовка данных успешно завершена!")


if __name__ == "__main__":
    main()