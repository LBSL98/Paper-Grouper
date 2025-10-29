"""
Controller for both Manual and Auto modes.
The GUI should call here, not core/io directly.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from paper_grouper.io.file_scanner import list_pdfs
from paper_grouper.core.metadata_extractor import batch_extract
from paper_grouper.core.embedder import embed_articles_light, embed_articles_model
from paper_grouper.core.graph_builder import build_knn_graph
from paper_grouper.core.community_detector import detect_communities_louvain
from paper_grouper.core.cluster_postprocess import finalize_clustering
from paper_grouper.core.autotune import run_autotune
from paper_grouper.core.scoring import summarize_for_autotune
from paper_grouper.io.output_writer import prepare_output_dir, write_clustered_files
from paper_grouper.io.report_writer import write_reports
from paper_grouper.io.graph_visualizer import render_graph_png
from paper_grouper.core.data import ArticleRecord, ClusteringResult, AutoTuneTrialResult


def run_manual(input_dir: str,
               output_dir: Optional[str],
               k: int,
               resolution: float,
               min_cluster_size: int,
               rename_with_title: bool) -> Dict[str, Any]:

    pdfs = list_pdfs(input_dir)
    articles_list = batch_extract(pdfs)
    articles_by_id = {a.id: a for a in articles_list}

    # desenvolvimento: rápido, sem torch
    emb = embed_articles_light(articles_list)
    # produção futura (quando quiser rodar embeddings reais):
    # emb = embed_articles_model(articles_list)
    G = build_knn_graph(emb, k=k)
    raw_part = detect_communities_louvain(G, resolution=resolution)

    clustering = finalize_clustering(
        raw_article_to_cluster=raw_part,
        G=G,
        articles=articles_list,
        min_cluster_size=min_cluster_size,
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
    )

    out_root = prepare_output_dir(input_dir, output_dir)
    write_clustered_files(out_root, clustering, articles_by_id, rename_with_title)
    write_reports(out_root, clustering, articles_by_id, trials_info=None)
    graph_png = render_graph_png(G, clustering, out_root)

    summary = summarize_for_autotune(clustering)

    return {
        "output_root": str(out_root),
        "graph_png": str(graph_png),
        "clustering": clustering,
        "summary": summary,
        "articles": articles_by_id,
        "autotune_trials": None,
    }


def run_auto(input_dir: str,
             output_dir: Optional[str],
             k_values: List[int],
             resolutions: List[float],
             min_cluster_sizes: List[int],
             max_workers: int,
             rename_with_title: bool) -> Dict[str, Any]:

    pdfs = list_pdfs(input_dir)
    articles_list = batch_extract(pdfs)
    articles_by_id = {a.id: a for a in articles_list}

    # desenvolvimento: rápido, sem torch
    emb = embed_articles_light(articles_list)
    # produção futura (quando quiser rodar embeddings reais):
    # emb = embed_articles_model(articles_list)

    best_cr, best_cfg, trials = run_autotune(
        articles=articles_list,
        emb=emb,
        k_values=k_values,
        resolutions=resolutions,
        min_cluster_sizes=min_cluster_sizes,
        max_workers=max_workers,
    )

    # rebuild graph for visualization using best k
    G_best = build_knn_graph(emb, k=int(best_cfg["k"]))
    out_root = prepare_output_dir(input_dir, output_dir)
    graph_png = render_graph_png(G_best, best_cr, out_root)

    write_clustered_files(out_root, best_cr, articles_by_id, rename_with_title)
    write_reports(out_root, best_cr, articles_by_id, trials_info=trials)

    summary = summarize_for_autotune(best_cr)

    return {
        "output_root": str(out_root),
        "graph_png": str(graph_png),
        "clustering": best_cr,
        "summary": summary,
        "best_cfg": best_cfg,
        "articles": articles_by_id,
        "autotune_trials": trials,
    }
