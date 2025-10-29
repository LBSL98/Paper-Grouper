from pathlib import Path
from typing import Dict
import shutil
from slugify import slugify
from paper_grouper.core.data import ArticleRecord, ClusteringResult


def _safe_filename(base: str, ext: str = ".pdf") -> str:
    s = slugify(base)[:120]
    if not s:
        s = "paper"
    if not s.endswith(ext.replace(".", "")):
        return f"{s}{ext}"
    return s


def prepare_output_dir(input_dir: str, desired_out: str | None = None) -> Path:
    in_path = Path(input_dir).resolve()
    if desired_out:
        out_root = Path(desired_out).resolve()
    else:
        cand = Path(str(in_path) + "_grouped")
        idx = 2
        while cand.exists():
            cand = Path(str(in_path) + f"_grouped_{idx}")
            idx += 1
        out_root = cand
    out_root.mkdir(parents=True, exist_ok=False)
    return out_root


def write_clustered_files(output_root: Path,
                          clustering: ClusteringResult,
                          articles: Dict[str, ArticleRecord],
                          rename_with_title: bool) -> None:
    for cid, members in clustering.clusters.items():
        label = clustering.cluster_labels.get(cid, f"cluster_{cid}")
        sub = output_root / f"{cid:02d}_{slugify(label)[:40]}"
        sub.mkdir(parents=True, exist_ok=True)

        used_names = set()
        for art_id in members:
            art = articles[art_id]
            src = Path(art.src_path)
            if rename_with_title and art.title:
                base = f"{art.year or ''} {art.title}".strip()
            else:
                base = src.stem
            candidate = _safe_filename(base)
            new_name = candidate
            c = 1
            while new_name in used_names:
                new_name = candidate.replace(".pdf", f"_{c}.pdf")
                c += 1
            used_names.add(new_name)
            dst = sub / new_name
            shutil.copy2(src, dst)
