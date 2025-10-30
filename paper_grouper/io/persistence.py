from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

APP_DIR = Path.home() / ".paper_grouper"
APP_DIR.mkdir(parents=True, exist_ok=True)

CFG_FILE = APP_DIR / "settings.json"
CACHE_DIR = APP_DIR / "emb_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_settings(defaults: dict[str, Any]) -> dict[str, Any]:
    if CFG_FILE.exists():
        try:
            data = json.loads(CFG_FILE.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in data.items() if k in defaults})
        except Exception:
            pass
    return defaults


def save_settings(settings: dict[str, Any]) -> None:
    TMP = CFG_FILE.with_suffix(".tmp")
    TMP.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    TMP.replace(CFG_FILE)


def _key_for_text(text: str, dim: int, algo: str = "light") -> str:
    h = hashlib.sha256()
    h.update(algo.encode())
    h.update(str(dim).encode())
    h.update(text.encode(errors="ignore"))
    return h.hexdigest()


def load_embedding_from_cache(text: str, dim: int, algo: str = "light"):
    key = _key_for_text(text, dim, algo)
    f = CACHE_DIR / f"{key}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            pass
    return None


def save_embedding_to_cache(text: str, vec, dim: int, algo: str = "light"):
    key = _key_for_text(text, dim, algo)
    f = CACHE_DIR / f"{key}.pkl"
    f.write_bytes(pickle.dumps(vec))
