import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("metrics")
LOG_FILE = LOG_DIR / "latency_log.jsonl"


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(row: Dict[str, Any], file_path: Optional[Path] = None) -> None:
    ensure_dir()
    path = file_path or LOG_FILE
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")