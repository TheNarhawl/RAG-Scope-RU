import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
EVAL_DIR = DATA_DIR / "eval"

HF_DATASET_URL = (
    "https://huggingface.co/datasets/FractalGPT/RRNCBPublic/resolve/main/rrncb_public_dataset.csv"
)

CSV_PATH = EVAL_DIR / "dataset_sample.csv"
OUT_PATH = EVAL_DIR / "questions.json"


def download_eval_dataset():
    """Скачивает CSV датасета RRNCB с Hugging Face."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print("==> 1. Скачивание датасета с Hugging Face...")
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

    print("==> 2. Конвертация вопросов в JSON...")
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


def unpack_or_verify_pdfs(custom_zip: Path | None = None):
    """
    Проверяет наличие PDF в data/docs/.
    Если их нет, ищет zip-архив (docs.zip или указанный пользователем) и распаковывает.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    existing_pdfs = list(DOCS_DIR.glob("*.pdf"))

    if existing_pdfs and not custom_zip:
        print(f"==> 3. PDF-документы уже на месте: найдено {len(existing_pdfs)} файлов в {DOCS_DIR}")
        return

    # Ищем архив
    zip_candidates = [
        custom_zip,
        DATA_DIR / "docs.zip",
        ROOT / "docs.zip",
    ]
    zip_file = next((p for p in zip_candidates if p and p.is_file()), None)

    if zip_file:
        print(f"==> 3. Распаковка архива {zip_file.name} в {DOCS_DIR}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            # Извлекаем только PDF-файлы, убирая лишние промежуточные папки
            extracted_count = 0
            for member in zf.infolist():
                if member.filename.lower().endswith(".pdf") and not member.is_dir():
                    filename = Path(member.filename).name
                    target_path = DOCS_DIR / filename
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted_count += 1

        print(f"    Успешно извлечено {extracted_count} PDF-файлов.")
    else:
        print("\n[ВНИМАНИЕ] PDF-файлы не найдены.")
        print(f"Пожалуйста, положите архив с документами 'docs.zip' в папку: {DATA_DIR}")
        print(f"Либо напрямую скопируйте PDF-файлы в папку: {DOCS_DIR}\n")


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для RAG-Scope-RU")
    parser.add_argument("--skip-pdfs", action="store_true", help="Пропустить проверку и распаковку PDF")
    parser.add_argument("--skip-questions", action="store_true", help="Пропустить скачивание и конвертацию вопросов")
    parser.add_argument("--zip", type=str, default=None, help="Путь к кастомному zip-архиву с PDF")
    args = parser.parse_args()

    if not args.skip_questions:
        download_eval_dataset()
        convert_questions()

    if not args.skip_pdfs:
        custom_zip = Path(args.zip) if args.zip else None
        unpack_or_verify_pdfs(custom_zip)

    print("\nГотово!")


if __name__ == "__main__":
    main()