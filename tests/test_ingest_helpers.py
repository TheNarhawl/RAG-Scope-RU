import sys
import uuid
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.ingest import make_point_id  # noqa: E402


def test_make_point_id_determinism():
    id1 = make_point_id("doc_a.pdf", page=1, chunk_id=0)
    id2 = make_point_id("doc_a.pdf", page=1, chunk_id=0)
    assert id1 == id2


def test_make_point_id_uniqueness():
    base = make_point_id("doc_a.pdf", page=1, chunk_id=0)

    diff_chunk = make_point_id("doc_a.pdf", page=1, chunk_id=1)
    diff_page = make_point_id("doc_a.pdf", page=2, chunk_id=0)
    diff_source = make_point_id("doc_b.pdf", page=1, chunk_id=0)

    assert base != diff_chunk
    assert base != diff_page
    assert base != diff_source
    assert len({base, diff_chunk, diff_page, diff_source}) == 4


def test_make_point_id_valid_uuid5():
    res = make_point_id("gost_12345.pdf", page=10, chunk_id=3)
    parsed = uuid.UUID(res)
    assert parsed.version == 5
