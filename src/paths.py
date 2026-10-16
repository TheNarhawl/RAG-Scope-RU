import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DOCS_DIR = DATA_DIR / "docs"
EVAL_DIR = DATA_DIR / "eval"
EXPERIMENTS_DIR = Path(os.getenv("EXPERIMENTS_DIR", ROOT / "experiments"))
CONFIGS_DIR = ROOT / "configs"

load_dotenv(ROOT / ".env")

QDRANT_URL = os.environ["QDRANT_URL"]

EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))