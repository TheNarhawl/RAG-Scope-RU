from pathlib import Path
import sys
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    top_k = cfg["retrieve"]["top_k"]

    query = " ".join(sys.argv[1:]) or "о чём этот документ?"

    model = SentenceTransformer(ing["embedding_model"])
    client = chromadb.PersistentClient(path=str(ROOT / "data" / "index"))
    col = client.get_collection(ing["collection"])

    q_emb = model.encode([query]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=top_k)

    print(f"Запрос: {query}\n")
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(f"--- {meta['source']} #{meta['chunk_id']} ---")
        print(doc[:400], "\n")


if __name__ == "__main__":
    main()