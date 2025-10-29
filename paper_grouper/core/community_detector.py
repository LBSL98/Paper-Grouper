"""
Community detection using Louvain.
"""

from typing import Dict

import networkx as nx
from community import community_louvain  # python-louvain


def detect_communities_louvain(G: nx.Graph, resolution: float) -> Dict[str, int]:
    """
    Returns mapping: article_id -> cluster_id
    """
    partition = community_louvain.best_partition(
        G,
        resolution=resolution,
        weight="weight",
    )
    return partition
