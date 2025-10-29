from paper_grouper.core.data import ClusteringResult
from paper_grouper.core.scoring import summarize_for_autotune


def test_summarize_for_autotune_minimal():
    dummy = ClusteringResult(
        article_to_cluster={"a":0,"b":0,"c":1},
        clusters={0:["a","b"],1:["c"]},
        cluster_labels={0:"x",1:"y"},
        modularity=0.9,
        balance_score=0.4,
        small_cluster_fraction=0.5,
        score_final=1.23,
        centrality={"a":1.0,"b":0.5,"c":0.1},
    )
    s = summarize_for_autotune(dummy)
    assert "score_final" in s
    assert s["n_clusters"] == 2
