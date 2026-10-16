from pathlib import Path
import gc
import time

import torch
import yaml
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ["QDRANT_PORT"])
EMBED_BATCH_SIZE = int(os.environ["EMBED_BATCH_SIZE"])

def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Возвращает [(номер_страницы, текст), ...]"""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def pick_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: cuda ({name}, {vram:.1f} GB VRAM)")
        return "cuda"
    print("Device: cpu (CUDA недоступна)")
    return "cpu"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]
    collection_name = ing["collection"]

    docs_dir = ROOT / "data" / "docs"
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Нет PDF в {docs_dir}.")
        return

    print(f"Найдено PDF: {len(pdfs)}")

    device = pick_device()
    model = SentenceTransformer(ing["embedding_model"], device=device)
    
    # Получаем размерность вектора из модели
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Размерность вектора: {vector_size}")

    # Подключаемся к Qdrant и создаём коллекцию
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Удаляем старую коллекцию, если есть, чтобы начать с чистого листа
    client.delete_collection(collection_name)  # Игнорируем ошибку, если коллекции нет
    print(f"Коллекция '{collection_name}' удалена (если существовала)")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Коллекция '{collection_name}' создана")

    total_points = 0
    skipped = []
    t0 = time.time()

    for idx, pdf in enumerate(pdfs, start=1):
        pdf_start = time.time()
        pages = load_pdf_pages(pdf)
        points = []

        for page_num, page_text in pages:
            for local_i, chunk in enumerate(
                chunk_text(page_text, ing["chunk_size"], ing["chunk_overlap"])
            ):
                # Формируем точку (PointStruct)
                point_id = f"{pdf.stem}_p{page_num}_c{local_i}"
                points.append({
                    "id": point_id,
                    "text": chunk,
                    "source": pdf.name,
                    "page": page_num,
                    "chunk_id": local_i,
                })

        if not points:
            skipped.append(pdf.name)
            print(f"[{idx}/{len(pdfs)}] {pdf.name}: пусто (нет текстового слоя)")
            continue

        # Вычисляем эмбеддинги для всех чанков этого PDF
        texts = [p["text"] for p in points]
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Формируем список PointStruct для Qdrant
        qdrant_points = [
            PointStruct(
                id=points[i]["id"],
                vector=embeddings[i].tolist(),
                payload={
                    "text": points[i]["text"],
                    "source": points[i]["source"],
                    "page": points[i]["page"],
                    "chunk_id": points[i]["chunk_id"],
                },
            )
            for i in range(len(points))
        ]

        # Загружаем точки через upload_points (сам управляет батчами)
        client.upload_points(
            collection_name=collection_name,
            points=qdrant_points,
            batch_size=256,   # рекомендуемый размер для стабильности[reference:3]
            parallel=1,
            wait=True,
        )

        total_points += len(qdrant_points)
        elapsed = time.time() - pdf_start
        print(
            f"[{idx}/{len(pdfs)}] {pdf.name}: {len(qdrant_points)} точек "
            f"из {len(pages)} стр. ({elapsed:.1f} сек)"
        )

    total_time = time.time() - t0

    # Освобождаем VRAM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Информация о коллекции
    info = client.get_collection(collection_name)
    print("\n=== Итог ===")
    print(f"Точек в коллекции: {info.points_count}")
    print(f"Добавлено в этом прогоне: {total_points}")
    print(f"Время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
    if skipped:
        print(f"Пропущено (пусто): {len(skipped)}")
        for name in skipped:
            print(f"  - {name}")


if __name__ == "__main__":
    main()