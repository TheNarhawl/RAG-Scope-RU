import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from paths import CONFIGS_DIR, QDRANT_URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks from Qdrant.")
    parser.add_argument(
        "--config",
        default="baseline",
        help="Имя конфига в configs/ без .yaml (по умолчанию baseline).",
    )
    parser.add_argument("query", nargs="*", help="Текст запроса.")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    top_k = cfg["retrieve"]["top_k"]

    query = " ".join(args.query) if args.query else "о чём этот документ?"

    model = SentenceTransformer(ing["embedding_model"])
    client = QdrantClient(url=QDRANT_URL)

    query_text = query
    if "e5" in ing["embedding_model"].lower():
        query_text = "query: " + query

    q_emb = model.encode([query_text], normalize_embeddings=True)[0].tolist()

    results = client.query_points(
        collection_name=ing["collection"],
        query=q_emb,
        limit=top_k,
        with_payload=True,
    )

    print(f"Конфиг: {args.config}")
    print(f"Коллекция: {ing['collection']}")
    print(f"Модель: {ing['embedding_model']}")
    print(f"Запрос: {query}\n")

    for hit in results.points:
        payload = hit.payload
        print(
            f"--- {payload['source']}, стр. {payload['page']} "
            f"(chunk {payload['chunk_id']}) score={hit.score:.4f} ---"
        )
        print(payload["text"][:400], "\n")


if __name__ == "__main__":
    main()