import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in (str(SRC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.utils import ensure_ragas_compatibility  # noqa: E402

# Автоматически регистрируем совместимость перед запуском любых тестов
ensure_ragas_compatibility()
