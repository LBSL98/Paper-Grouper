from pathlib import Path
from typing import Dict, List, Optional
import json
from paper_grouper.core.data import ArticleRecord, ClusteringResult, AutoTuneTrialResult


def write_reports(output_root: Path,
                  clustering: ClusteringResult,
                  articles: Dict[str, ArticleRecord],
                  trials_info: Optional[List[AutoTuneTrialResult]] = None) -> None:

    # JSON
    json_path = output_root / "clusters_summary.json"
    data = {
        "score_final": clustering.score_final,
        "clusters": [],
    }

    for cid, members in clustering.clusters.items():
        cinfo = {
            "cluster_id": cid,
            "label": clustering.cluster_labels.get(cid, f"cluster_{cid}"),
            "size": len(members),
            "papers": [],
        }
        for art_id in members:
            a = articles[art_id]
            cinfo["papers"].append({
                "id": a.id,
                "title": a.title,
                "year": a.year,
                "abstract_snippet": a.abstract[:400],
            })
        data["clusters"].append(cinfo)

    if trials_info:
        data["autotune_trials"] = [
            {
                "params": t.params,
                "score_final": t.score_final,
                "n_clusters": t.n_clusters,
                "max_cluster_fraction": t.max_cluster_fraction,
            }
            for t in trials_info
        ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # TXT
    txt_path = output_root / "clusters_overview.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Final score: {clustering.score_final:.4f}\n\n")
        for cid, members in clustering.clusters.items():
            label = clustering.cluster_labels.get(cid, f"cluster_{cid}")
            f.write(f"=== Cluster {cid} :: {label} (size={len(members)}) ===\n")
            ranked = sorted(
                members,
                key=lambda aid: clustering.centrality.get(aid, 0.0),
                reverse=True,
            )
            for aid in ranked:
                a = articles[aid]
                f.write(f"- {a.title} ({a.year}) [{aid}]\n")
            f.write("\n")
