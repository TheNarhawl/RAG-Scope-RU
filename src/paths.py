import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
DOCS_DIR = DATA_DIR / "docs"
EVAL_DIR = DATA_DIR / "eval"
CONFIGS_DIR = ROOT / "configs"
EXPERIMENTS_DIR = Path(os.getenv("EXPERIMENTS_DIR", str(ROOT / "experiments")))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

from constants import DEFAULT_EMBED_BATCH_SIZE

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE)))
