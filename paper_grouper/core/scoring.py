from .data import ClusteringResult


def summarize_for_autotune(cr: ClusteringResult) -> dict:
    total = sum(len(v) for v in cr.clusters.values())
    max_cluster = max(len(v) for v in cr.clusters.values())
    return {
        "n_clusters": len(cr.clusters),
        "max_cluster_fraction": max_cluster / max(1, total),
        "modularity": cr.modularity,
        "balance_score": cr.balance_score,
        "small_cluster_fraction": cr.small_cluster_fraction,
        "score_final": cr.score_final,
    }
