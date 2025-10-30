# paper_grouper/core/embedder.py
from __future__ import annotations

import hashlib

import numpy as np

from ..io.persistence import (
    load_embedding_from_cache,
    save_embedding_to_cache,
)
from .data import ArticleRecord, EmbeddingResult


def _text_to_vec_hash(text: str, dim: int = 64) -> np.ndarray:
    """
    Vetor leve e determinístico via hashing (para rodar offline, sem modelos pesados).
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(text.encode("utf-8", errors="ignore"))
    raw = h.digest()

    # repete os bytes até preencher 'dim'
    buf = (raw * ((dim // len(raw)) + 1))[:dim]
    arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    arr = arr / max(1.0, float(arr.std() or 1.0))  # normaliza rudimentarmente
    return arr


def embed_articles_light(
    articles: list[ArticleRecord], dim: int = 64
) -> EmbeddingResult:
    """
    Gera embeddings leves (hash) para uma lista de ArticleRecord.
    Usa cache em disco para acelerar reexecuções.
    """
    ids: list[str] = []
    vectors: list[np.ndarray] = []

    for a in articles:
        ids.append(a.id)
        cached = load_embedding_from_cache(a.text_repr, dim, "light")
        if cached is not None:
            vectors.append(cached)
            continue
        v = _text_to_vec_hash(a.text_repr, dim)
        save_embedding_to_cache(a.text_repr, v, dim, "light")
        vectors.append(v)

    return EmbeddingResult(vectors=np.vstack(vectors), article_ids=ids)
