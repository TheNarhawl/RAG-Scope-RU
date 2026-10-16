import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ["QDRANT_PORT"])


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    top_k = cfg["retrieve"]["top_k"]

    query = " ".join(sys.argv[1:]) or "о чём этот документ?"

    model = SentenceTransformer(ing["embedding_model"])
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = client.search(
        collection_name=ing["collection"],
        query_vector=q_emb,
        limit=top_k,
        with_payload=True,
    )

    print(f"Запрос: {query}\n")
    for hit in results:
        payload = hit.payload
        print(f"--- {payload['source']}, стр. {payload['page']} (chunk {payload['chunk_id']}) score={hit.score:.4f} ---")
        print(payload['text'][:400], "\n")


if __name__ == "__main__":
    main()