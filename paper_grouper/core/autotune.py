import concurrent.futures
import itertools
from typing import Dict, List, Tuple

from .cluster_postprocess import finalize_clustering
from .community_detector import detect_communities_louvain
from .data import (
    ArticleRecord,
    AutoTuneTrialResult,
    ClusteringResult,
    EmbeddingResult,
)
from .graph_builder import build_knn_graph
from .scoring import summarize_for_autotune


def _evaluate_single_config(
    config: Dict[str, float],
    articles: List[ArticleRecord],
    emb: EmbeddingResult,
    alpha_beta_gamma=(1.0, 0.5, 0.5),
) -> Tuple[Dict[str, float], ClusteringResult, Dict[str, float]]:
    k = int(config["k"])
    resolution = float(config["resolution"])
    min_cluster = int(config["min_cluster_size"])

    G = build_knn_graph(emb, k=k)
    raw_part = detect_communities_louvain(G, resolution=resolution)

    cr = finalize_clustering(
        raw_article_to_cluster=raw_part,
        G=G,
        articles=articles,
        min_cluster_size=min_cluster,
        alpha=alpha_beta_gamma[0],
        beta=alpha_beta_gamma[1],
        gamma=alpha_beta_gamma[2],
    )

    summary = summarize_for_autotune(cr)
    return (config, cr, summary)


def run_autotune(
    articles: List[ArticleRecord],
    emb: EmbeddingResult,
    k_values: List[int],
    resolutions: List[float],
    min_cluster_sizes: List[int],
    max_workers: int = 4,
) -> Tuple[ClusteringResult, Dict[str, float], List[AutoTuneTrialResult]]:

    configs = []
    for k, r, m in itertools.product(k_values, resolutions, min_cluster_sizes):
        configs.append(
            {
                "k": k,
                "resolution": r,
                "min_cluster_size": m,
            }
        )

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(_evaluate_single_config, cfg, articles, emb) for cfg in configs
        ]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    best = max(results, key=lambda r: r[2]["score_final"])
    best_config, best_cr, best_summary = best

    trials: List[AutoTuneTrialResult] = []
    for cfg, _cr, summary in results:
        trials.append(
            AutoTuneTrialResult(
                params=cfg,
                n_clusters=summary["n_clusters"],
                max_cluster_fraction=summary["max_cluster_fraction"],
                modularity=summary["modularity"],
                balance_score=summary["balance_score"],
                small_cluster_fraction=summary["small_cluster_fraction"],
                score_final=summary["score_final"],
            )
        )

    return best_cr, best_config, trials
