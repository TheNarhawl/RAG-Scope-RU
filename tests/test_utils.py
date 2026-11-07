import random
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import get_qdrant_client, pick_device, set_seed  # noqa: E402


def test_set_seed_reproducibility():
    set_seed(123)
    val_py1 = random.random()
    val_np1 = np.random.rand()

    set_seed(123)
    val_py2 = random.random()
    val_np2 = np.random.rand()

    assert val_py1 == val_py2
    assert val_np1 == val_np2


def test_pick_device():
    device = pick_device(verbose=False)
    assert device in ("cpu", "cuda")


def test_get_qdrant_client():
    pytest.importorskip("qdrant_client")
    client = get_qdrant_client("http://localhost:6333")
    assert client is not None
