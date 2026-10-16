import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "eval" / "dataset_sample.csv"
OUT_PATH = ROOT / "data" / "eval" / "questions.json"

def convert():
    questions = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            questions.append({
                "id": f"q{i:03d}",
                "question": row["question"],
                "ground_truth": row["answer"],
                "source_hint": row["document"]
            })
    
    OUT_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено {len(questions)} вопросов в {OUT_PATH}")

if __name__ == "__main__":
    convert()