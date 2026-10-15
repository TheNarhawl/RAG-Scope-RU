from pathlib import Path
import yaml
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    ing = cfg["ingest"]

    docs_dir = ROOT / "data" / "docs"
    db_path = ROOT / "data" / "index"

    model = SentenceTransformer(ing["embedding_model"])
    client = chromadb.PersistentClient(path=str(db_path))
    col = client.get_or_create_collection(ing["collection"])

    pdfs = list(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Нет PDF в {docs_dir}. Положи туда хотя бы один файл.")
        return

    for pdf in pdfs:
        text = load_pdf(pdf)
        chunks = chunk_text(text, ing["chunk_size"], ing["chunk_overlap"])
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        ids = [f"{pdf.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf.name, "chunk_id": i} for i in range(len(chunks))]
        col.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        print(f"{pdf.name}: {len(chunks)} чанков")


if __name__ == "__main__":
    main()