"""
Post-process partition:
- merge tiny clusters
- compute metrics (modularity, balance, etc.)
- label clusters
- compute centrality
- score final solution
"""

from typing import Dict, List
import networkx as nx
from collections import Counter, defaultdict
from community import community_louvain
from .data import ArticleRecord, ClusteringResult


def _invert_partition(article_to_cluster: Dict[str, int]) -> Dict[int, List[str]]:
    clusters: Dict[int, List[str]] = defaultdict(list)
    for art, cid in article_to_cluster.items():
        clusters[cid].append(art)
    return clusters


def _merge_tiny_clusters(article_to_cluster: Dict[str, int],
                         G: nx.Graph,
                         min_size: int) -> Dict[str, int]:
    clusters = _invert_partition(article_to_cluster)
    tiny = {cid for cid, members in clusters.items() if len(members) < min_size}
    if not tiny:
        return article_to_cluster
    new_assign = dict(article_to_cluster)
    for cid in tiny:
        for art in clusters[cid]:
            best_cid = None
            best_w = -1.0
            for nbr in G.neighbors(art):
                nbr_cid = article_to_cluster[nbr]
                if nbr_cid == cid:
                    continue
                w = G[art][nbr].get("weight", 0.0)
                if w > best_w:
                    best_w = w
                    best_cid = nbr_cid
            if best_cid is not None:
                new_assign[art] = best_cid
    return new_assign


def _compute_centrality(G: nx.Graph,
                        clusters: Dict[int, List[str]]) -> Dict[str, float]:
    centrality: Dict[str, float] = {}
    for cid, members in clusters.items():
        member_set = set(members)
        for m in members:
            score = 0.0
            for nbr in G[m]:
                if nbr in member_set:
                    score += G[m][nbr].get("weight", 0.0)
            centrality[m] = score
    return centrality


def _label_cluster(cid: int,
                   members: List[str],
                   by_id: Dict[str, ArticleRecord]) -> str:
    bag = []
    for art_id in members:
        a = by_id[art_id]
        bag.extend(a.title.lower().split())
        bag.extend(a.abstract.lower().split())
    stop = {
        "the","a","an","and","of","for","to","in","on","with","using",
        "um","uma","de","da","do","para","em"
    }
    bag = [
        w.strip(".,:;()[]") for w in bag
        if w.lower() not in stop and len(w) > 2
    ]
    common = [w for (w, _) in Counter(bag).most_common(4)]
    return " / ".join(common) if common else f"cluster_{cid}"


def _balance_score(clusters: Dict[int, List[str]], total_n: int) -> float:
    max_size = max(len(v) for v in clusters.values())
    return 1.0 - (max_size / float(total_n))


def _small_frac(clusters: Dict[int, List[str]], min_size: int) -> float:
    tiny_count = sum(1 for v in clusters.values() if len(v) < min_size)
    return tiny_count / max(1, len(clusters))


def finalize_clustering(raw_article_to_cluster: Dict[str, int],
                        G: nx.Graph,
                        articles: List[ArticleRecord],
                        min_cluster_size: int,
                        alpha: float,
                        beta: float,
                        gamma: float) -> ClusteringResult:

    reassigned = _merge_tiny_clusters(raw_article_to_cluster, G, min_cluster_size)
    clusters = _invert_partition(reassigned)

    modularity = community_louvain.modularity(reassigned, G, weight="weight")

    total_n = len(reassigned)
    balance_score = _balance_score(clusters, total_n)
    small_fraction = _small_frac(clusters, min_cluster_size)

    by_id = {a.id: a for a in articles}
    cluster_labels = {
        cid: _label_cluster(cid, members, by_id)
        for cid, members in clusters.items()
    }

    centrality = _compute_centrality(G, clusters)

    score_final = alpha * modularity + beta * balance_score - gamma * small_fraction

    return ClusteringResult(
        article_to_cluster=reassigned,
        clusters=clusters,
        cluster_labels=cluster_labels,
        modularity=modularity,
        balance_score=balance_score,
        small_cluster_fraction=small_fraction,
        score_final=score_final,
        centrality=centrality,
    )
