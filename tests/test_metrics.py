import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.metrics import hit_at_k, rr, slugify  # noqa: E402


def test_hit_at_k():
    sources = ["docA.pdf", "docB.pdf", "docC.pdf", "docD.pdf"]

    # Hit at k=1
    assert hit_at_k(sources, "docA.pdf", 1) == 1.0
    assert hit_at_k(sources, "docB.pdf", 1) == 0.0

    # Hit at k=3
    assert hit_at_k(sources, "docC.pdf", 3) == 1.0
    assert hit_at_k(sources, "docD.pdf", 3) == 0.0
    assert hit_at_k(sources, "docD.pdf", 4) == 1.0

    # Non-existent target
    assert hit_at_k(sources, "docZ.pdf", 10) == 0.0

    # Empty list
    assert hit_at_k([], "docA.pdf", 5) == 0.0


def test_reciprocal_rank():
    sources = ["docA.pdf", "docB.pdf", "docC.pdf", "docD.pdf"]

    assert rr(sources, "docA.pdf") == 1.0
    assert rr(sources, "docB.pdf") == 0.5
    assert rr(sources, "docC.pdf") == 1.0 / 3.0
    assert rr(sources, "docD.pdf") == 0.25

    # Target not in list
    assert rr(sources, "docZ.pdf") == 0.0
    assert rr([], "docA.pdf") == 0.0


def test_slugify():
    assert slugify("USER BGE-M3+Hybrid") == "user_bge_m3hybrid"
    assert slugify("test-run 1") == "test_run_1"
    assert slugify("Clean_Name") == "clean_name"
