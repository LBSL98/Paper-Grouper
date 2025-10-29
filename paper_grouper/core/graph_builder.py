"""
Build k-NN similarity graph using cosine similarity of embeddings.
"""

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data import EmbeddingResult


def build_knn_graph(emb: EmbeddingResult, k: int) -> nx.Graph:
    sims = cosine_similarity(emb.vectors)  # N x N
    n = sims.shape[0]

    G = nx.Graph()
    for node_id in emb.article_ids:
        G.add_node(node_id)

    for i in range(n):
        idx_sorted = np.argsort(-sims[i])
        neighbors = [j for j in idx_sorted if j != i][:k]
        for j in neighbors:
            a = emb.article_ids[i]
            b = emb.article_ids[j]
            w = float(sims[i, j])
            if a == b:
                continue
            if G.has_edge(a, b):
                if w > G[a][b]["weight"]:
                    G[a][b]["weight"] = w
            else:
                G.add_edge(a, b, weight=w)

    return G
