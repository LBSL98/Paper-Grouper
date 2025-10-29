from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class ArticleRecord:
    """Minimal structured representation of a paper/article."""
    id: str                # unique ID, usually original filename
    src_path: str          # absolute path to original PDF
    title: str
    abstract: str
    keywords: str
    year: Optional[int]
    text_repr: str         # concatenation of title+abstract+keywords (cleaned)


@dataclass
class EmbeddingResult:
    """Stores embeddings for all articles."""
    vectors: np.ndarray           # shape (N, D)
    article_ids: List[str]        # len N, aligns with vectors rows


@dataclass
class ClusteringResult:
    """Stores final clustering and metrics for one parameter configuration."""
    article_to_cluster: Dict[str, int]      # article_id -> cluster_id
    clusters: Dict[int, List[str]]          # cluster_id -> [article_id,...]
    cluster_labels: Dict[int, str]          # cluster_id -> human-readable label
    modularity: float                       # partition modularity score
    balance_score: float                    # 1 - (max_cluster_size / total)
    small_cluster_fraction: float           # frac of clusters under threshold
    score_final: float                      # combined score we optimize
    centrality: Dict[str, float]            # article_id -> importance within cluster


@dataclass
class AutoTuneTrialResult:
    """One trial in autotuning space."""
    params: Dict[str, float]
    n_clusters: int
    max_cluster_fraction: float
    modularity: float
    balance_score: float
    small_cluster_fraction: float
    score_final: float
