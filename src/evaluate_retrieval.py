import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from paths import CONFIGS_DIR, EVAL_DIR, EXPERIMENTS_DIR, QDRANT_URL

KS = [1, 3, 5, 10]


def short_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def slugify(text: str) -> str:
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
    parser.add_argument("--tag", default=None, help="Тег прогона.")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число вопросов.")
    parser.add_argument("--config", default="baseline", help="Имя конфига в configs/.")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]

    questions = load_questions(EVAL_DIR / "questions.json")
    if not questions:
        print("questions.json пуст.")
        return

    if args.limit:
        questions = questions[: args.limit]
        print(f"(взяты первые {args.limit} вопросов)")

    model = SentenceTransformer(ing["embedding_model"])
    client = QdrantClient(url=QDRANT_URL)

    info = client.get_collection(ing["collection"])
    print(f"Конфиг: {args.config}")
    print(f"Коллекция: {ing['collection']}, точек: {info.points_count}")
    print(f"Модель: {ing['embedding_model']}")
    print(f"Вопросов: {len(questions)}\n")

    is_e5 = "e5" in ing["embedding_model"].lower()
    max_k = max(KS)
    hits = {k: 0.0 for k in KS}
    mrr_sum = 0.0
    per_question = []

    t0 = time.time()

    for q in questions:
        q_text = q["question"]
        if is_e5:
            q_text = "query: " + q_text
        q_emb = model.encode([q_text], normalize_embeddings=True)[0].tolist()

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

    elapsed = time.time() - t0
    n = len(questions)

    summary = {
        "n_questions": n,
        "hit_rate": {f"@{k}": hits[k] / n for k in KS},
        "mrr": mrr_sum / n,
        "elapsed_sec": elapsed,
        "ms_per_query": elapsed / n * 1000,
    }

    print("=== Метрики (по источнику) ===")
    for k in KS:
        print(f"  Hit Rate@{k}: {summary['hit_rate'][f'@{k}']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}")
    print(f"  Время:       {elapsed:.1f} сек ({summary['ms_per_query']:.1f} мс/вопрос)\n")

    print("=== Детали ===")
    for row in per_question:
        flags = " ".join(f"h@{k}={int(row[f'hit@{k}'])}" for k in KS)
        short_q = row["question"][:50]
        print(f"  {row['id']}: {flags} rr={row['rr']:.2f} | {short_q}")

    tag = args.tag or short_model_name(ing["embedding_model"])
    tag = slugify(tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{tag}_{timestamp}.json"

    out_dir = EXPERIMENTS_DIR / ing["collection"]
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