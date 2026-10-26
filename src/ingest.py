import argparse
import gc
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from fastembed import SparseTextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, Modifier, PointStruct, SparseVectorParams, VectorParams,
)
from sentence_transformers import SentenceTransformer

from paths import CONFIGS_DIR, DOCS_DIR, EMBED_BATCH_SIZE, QDRANT_URL
from splitters import naive_chunk_text, structural_chunk_text

UUID_NAMESPACE = uuid.UUID("6f2a3b1c-4d5e-4f6a-8b7c-9d0e1f2a3b4c")


def make_point_id(source: str, page: int, chunk_id: int) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, f"{source}|p{page}|c{chunk_id}"))


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def pick_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: cuda ({name}, {vram:.1f} GB VRAM)")
        return "cuda"
    print("Device: cpu")
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline")
    args = parser.parse_args()

    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    collection_name = ing["collection"]
    vector_name = cfg["retrieve"].get("vector_name", "dense")
    chunk_strategy = ing.get("chunk_strategy", "naive")

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Нет PDF в {DOCS_DIR}.")
        return

    print(f"Конфиг: {args.config}")
    print(f"Коллекция: {collection_name}")
    print(f"Модель: {ing['embedding_model']}")
    print(f"Стратегия чанкинга: {chunk_strategy}")
    print(f"Чанк: {ing['chunk_size']} / overlap: {ing['chunk_overlap']}")
    print(f"Найдено PDF: {len(pdfs)}")

    device = pick_device()
    dense_model = SentenceTransformer(ing["embedding_model"], device=device)
    dense_dim = dense_model.get_embedding_dimension()
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")

    client = QdrantClient(url=QDRANT_URL)
    try:
        client.delete_collection(collection_name)
        print(f"Коллекция '{collection_name}' удалена")
    except Exception:
        print(f"Коллекции '{collection_name}' не было")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={vector_name: VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"Коллекция '{collection_name}' создана (dense + sparse)\n")

    is_e5 = "e5" in ing["embedding_model"].lower()
    passage_prefix = ing.get("passage_prefix", "passage: " if is_e5 else "")
    total_points = 0
    skipped = []
    t0 = time.time()

    # Выбор алгоритма сплиттера
    split_fn = structural_chunk_text if chunk_strategy == "structural" else naive_chunk_text

    for idx, pdf in enumerate(pdfs, start=1):
        pdf_start = time.time()
        points = []
        for page_num, page_text in load_pdf_pages(pdf):
            chunks = split_fn(page_text, ing["chunk_size"], ing["chunk_overlap"])
            for local_i, chunk in enumerate(chunks):
                points.append({
                    "id": make_point_id(pdf.name, page_num, local_i),
                    "text": chunk,
                    "source": pdf.name,
                    "page": page_num,
                    "chunk_id": local_i,
                })

        if not points:
            skipped.append(pdf.name)
            print(f"[{idx}/{len(pdfs)}] {pdf.name}: пусто")
            continue

        texts = [p["text"] for p in points]
        dense_texts = [passage_prefix + t for t in texts] if passage_prefix else texts
        dense_embs = dense_model.encode(
            dense_texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        sparse_embs = list(sparse_model.embed(texts))

        qdrant_points = [
            PointStruct(
                id=p["id"],
                vector={
                    vector_name: dense_embs[i].tolist(),
                    "sparse": {
                        "indices": sparse_embs[i].indices.tolist(),
                        "values": sparse_embs[i].values.tolist(),
                    },
                },
                payload={
                    "text": p["text"], "source": p["source"],
                    "page": p["page"], "chunk_id": p["chunk_id"],
                },
            )
            for i, p in enumerate(points)
        ]

        client.upload_points(
            collection_name=collection_name, points=qdrant_points,
            batch_size=256, parallel=1, wait=True,
        )
        total_points += len(qdrant_points)
        print(
            f"[{idx}/{len(pdfs)}] {pdf.name}: {len(qdrant_points)} точек "
            f"({time.time() - pdf_start:.1f} сек)"
        )

    total_time = time.time() - t0
    del dense_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    info = client.get_collection(collection_name)
    print("\n=== Итог ===")
    print(f"Точек: {info.points_count}")
    print(f"Время: {total_time:.1f} сек ({total_time / 60:.1f} мин)")
    if skipped:
        print(f"Пропущено: {skipped}")


if __name__ == "__main__":
    main()