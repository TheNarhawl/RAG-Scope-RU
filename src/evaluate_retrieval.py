import json
from pathlib import Path

import chromadb
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KS = [1, 3, 5]

def hit_source(retrieved_metas: list[dict], q: dict, k: int) -> float:
    for m in retrieved_metas[:k]:
        if m["source"] == q["source_hint"]:
            return 1.0
    return 0.0

def reciprocal_rank(retrieved_metas: list[dict], q: dict) -> float:
    for rank, m in enumerate(retrieved_metas, start=1):
        if m["source"] == q["source_hint"]:
            return 1.0 / rank
    return 0.0

def main():
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    
    model = SentenceTransformer(cfg["ingest"]["embedding_model"])
    client = chromadb.PersistentClient(path=str(ROOT / "data" / "index"))
    col = client.get_collection(cfg["ingest"]["collection"])

    hits = {k: 0.0 for k in KS}
    mrr_sum = 0.0

    for q in questions:
        q_emb = model.encode([q["question"]]).tolist()
        res = col.query(query_embeddings=q_emb, n_results=max(KS))
        metas = res["metadatas"][0]
        
        for k in KS:
            hits[k] += hit_source(metas, q, k)
        mrr_sum += reciprocal_rank(metas, q)

    n = len(questions)
    print(f"Hit Rate@1: {hits[1]/n:.3f}")
    print(f"Hit Rate@3: {hits[3]/n:.3f}")
    print(f"Hit Rate@5: {hits[5]/n:.3f}")
    print(f"MRR: {mrr_sum/n:.3f}")

if __name__ == "__main__":
    main()