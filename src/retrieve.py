import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from paths import CONFIGS_DIR, QDRANT_URL

RERANKER_MODEL = "BAAI/bge-reranker-base"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline")
    p.add_argument("query", nargs="*")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    ret = cfg["retrieve"]

    query = " ".join(args.query) if args.query else "о чём этот документ?"
    top_k = ret["top_k"]
    vector_name = ret.get("vector_name", "dense")
    use_hybrid = ret.get("use_hybrid", False)
    use_rerank = ret.get("use_rerank", False)
    rerank_candidates = ret.get("rerank_candidates", 20)

    dense_model = SentenceTransformer(ing["embedding_model"])
    client = QdrantClient(url=QDRANT_URL)

    is_e5 = "e5" in ing["embedding_model"].lower()
    q_dense_text = "query: " + query if is_e5 else query
    q_dense = dense_model.encode([q_dense_text], normalize_embeddings=True)[0].tolist()

    if use_hybrid:
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
        q_sparse = list(sparse_model.embed([query]))[0]
        results = client.query_points(
            collection_name=ing["collection"],
            prefetch=[
                models.Prefetch(query=q_dense, using=vector_name, limit=rerank_candidates),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=q_sparse.indices.tolist(),
                        values=q_sparse.values.tolist(),
                    ),
                    using="sparse",
                    limit=rerank_candidates,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),
            limit=rerank_candidates if use_rerank else top_k,
            with_payload=True,
        )
    else:
        results = client.query_points(
            collection_name=ing["collection"],
            query=q_dense,
            using=vector_name,
            limit=top_k,
            with_payload=True,
        )

    hits = results.points

    if use_rerank:
        reranker = CrossEncoder(RERANKER_MODEL)
        pairs = [(query, h.payload["text"]) for h in hits]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        final = [(s, h) for s, h in ranked[:top_k]]
    else:
        final = [(h.score, h) for h in hits[:top_k]]

    print(f"Конфиг: {args.config}")
    print(f"Hybrid: {use_hybrid} | Rerank: {use_rerank}")
    print(f"Запрос: {query}\n")
    for score, h in final:
        p = h.payload
        print(f"--- {p['source']}, стр. {p['page']} (chunk {p['chunk_id']}) score={score:.4f} ---")
        print(p["text"][:400], "\n")


if __name__ == "__main__":
    main()