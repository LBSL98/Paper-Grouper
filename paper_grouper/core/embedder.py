"""
Embedding interface.

Durante desenvolvimento da UI, carregar sentence-transformers + torch
pode ficar pesado e travar a janela. Então oferecemos dois modos:

1. embed_articles_light(...)  -> rápido, não usa torch
   Gera vetores "fake" porém consistentes a partir do texto
   usando hashing simples. Bom pra testar fluxo.

2. embed_articles_model(...)  -> usa sentence-transformers
   (usa torch, pesado)

O controller pode chamar um ou outro.
"""

import hashlib
from typing import List

import numpy as np

from .data import ArticleRecord, EmbeddingResult


def _text_to_vec_hash(text: str, dim: int = 64) -> np.ndarray:
    """
    Gera um vetor fixo baseado em hashing de palavras.
    Ideia: cada palavra mapeia para um índice pseudo-aleatório no vetor
    e incrementa aquele índice.
    Isso NÃO é semântica de verdade, mas mantém artigos com palavras
    parecidas mais próximos que artigos totalmente diferentes.
    """
    vec = np.zeros(dim, dtype=float)
    for token in text.lower().split():
        # hash da palavra -> inteiro grande -> índice [0, dim)
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # normaliza para evitar que textos longos dominem demais
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_articles_light(
    articles: List[ArticleRecord], dim: int = 64
) -> EmbeddingResult:
    """
    Modo leve (sem torch). Útil para desenvolvimento rápido.
    """
    vectors = []
    ids = []
    for a in articles:
        v = _text_to_vec_hash(a.text_repr, dim=dim)
        vectors.append(v)
        ids.append(a.id)
    vectors = np.vstack(vectors)
    return EmbeddingResult(
        vectors=vectors,
        article_ids=ids,
    )


# --------- PESADO / REAL (desativado por padrão no controller agora) ----------

_model_cache = None


def _get_model():
    """
    Carrega sentence-transformers (usa torch). Só chamamos isso
    em modo 'real'.
    """
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer

        # modelo pequeno mas ainda usa torch
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def embed_articles_model(articles: List[ArticleRecord]) -> EmbeddingResult:
    """
    Modo real, usa embeddings semânticos de verdade.
    Isso é o que você vai usar mais tarde, mas requer PyTorch.
    """
    model = _get_model()
    texts = [a.text_repr for a in articles]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return EmbeddingResult(
        vectors=np.asarray(vectors, dtype=float),
        article_ids=[a.id for a in articles],
    )
