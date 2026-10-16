import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ["QDRANT_PORT"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks from Qdrant.")
    parser.add_argument(
        "--config",
        default="baseline",
        help="Имя конфига в configs/ без .yaml (по умолчанию baseline).",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Текст запроса.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    top_k = cfg["retrieve"]["top_k"]

    query = " ".join(args.query) if args.query else "о чём этот документ?"

    model = SentenceTransformer(ing["embedding_model"])
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

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