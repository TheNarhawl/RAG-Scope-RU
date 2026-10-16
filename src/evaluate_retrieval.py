import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ["QDRANT_PORT"])

KS = [1, 3, 5]


def short_model_name(model_id: str) -> str:
    """'sentence-transformers/all-MiniLM-L6-v2' -> 'all-MiniLM-L6-v2'"""
    return model_id.split("/")[-1]


def slugify(text: str) -> str:
    """'Qdrant + MiniLM' -> 'qdrant_minilm'"""
    return text.lower().replace(" ", "_").replace("+", "").replace("-", "_")


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def hit_at_k(retrieved_sources: list[str], target: str, k: int) -> float:
    return 1.0 if target in retrieved_sources[:k] else 0.0


def reciprocal_rank(retrieved_sources: list[str], target: str) -> float:
    for rank, src in enumerate(retrieved_sources, start=1):
        if src == target:
            return 1.0 / rank
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on Qdrant.")
    parser.add_argument(
        "--tag",
        default=None,
        help="Тег прогона для имени файла. По умолчанию — короткое имя embedding-модели.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число вопросов (для быстрой проверки).",
    )
    parser.add_argument(
        "--config",
        default="baseline",
        help="Имя конфига в configs/ без .yaml (по умолчанию baseline).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]

    questions = load_questions(ROOT / "data" / "eval" / "questions.json")
    if not questions:
        print("questions.json пуст.")
        return

    if args.limit:
        questions = questions[: args.limit]
        print(f"(взяты первые {args.limit} вопросов)")

    model = SentenceTransformer(ing["embedding_model"])
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    info = client.get_collection(ing["collection"])
    print(f"Коллекция: {ing['collection']}, точек: {info.points_count}")
    print(f"Модель: {ing['embedding_model']}")
    print(f"Вопросов: {len(questions)}\n")

    max_k = max(KS)
    hits = {k: 0.0 for k in KS}
    mrr_sum = 0.0
    per_question = []

    for q in questions:
        q_emb = model.encode([q["question"]], normalize_embeddings=True)[0].tolist()

        res = client.query_points(
            collection_name=ing["collection"],
            query=q_emb,
            limit=max_k,
            with_payload=True,
        )

        retrieved_sources = [h.payload["source"] for h in res.points]
        target = q["source_hint"]

        row = {
            "id": q["id"],
            "question": q["question"],
            "target": target,
            "retrieved": retrieved_sources,
        }

        for k in KS:
            h = hit_at_k(retrieved_sources, target, k)
            hits[k] += h
            row[f"hit@{k}"] = h

        rr = reciprocal_rank(retrieved_sources, target)
        mrr_sum += rr
        row["rr"] = rr
        per_question.append(row)

    n = len(questions)
    summary = {
        "n_questions": n,
        "hit_rate": {f"@{k}": hits[k] / n for k in KS},
        "mrr": mrr_sum / n,
    }

    print("=== Метрики (по источнику) ===")
    for k in KS:
        print(f"  Hit Rate@{k}: {summary['hit_rate'][f'@{k}']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}\n")

    print("=== Детали ===")
    for row in per_question:
        flags = " ".join(f"h@{k}={int(row[f'hit@{k}'])}" for k in KS)
        short_q = row["question"][:55]
        print(f"  {row['id']}: {flags} rr={row['rr']:.2f} | {short_q}")

    # Формируем имя файла
    tag = args.tag or short_model_name(ing["embedding_model"])
    tag = slugify(tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{tag}_{timestamp}.json"

    out_dir = ROOT / "experiments" / ing["collection"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    payload = {
        "config": cfg,
        "tag": tag,
        "timestamp": timestamp,
        "summary": summary,
        "per_question": per_question,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()