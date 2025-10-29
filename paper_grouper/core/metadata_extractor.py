"""
Extract minimal metadata (title, abstract, keywords, year) from PDFs.
Right now it's a placeholder heuristic, later we'll parse PDF text
and/or use .bib.
"""

from typing import List
from pathlib import Path
from .data import ArticleRecord


def extract_from_pdf(pdf_path: str) -> ArticleRecord:
    p = Path(pdf_path)
    fake_title = p.stem
    fake_abs = ""
    fake_kw = ""
    fake_year = None

    text_repr = f"{fake_title}. {fake_abs}. {fake_kw}".strip()

    return ArticleRecord(
        id=p.name,
        src_path=str(p.resolve()),
        title=fake_title,
        abstract=fake_abs,
        keywords=fake_kw,
        year=fake_year,
        text_repr=text_repr,
    )


def batch_extract(pdf_paths: List[str]) -> List[ArticleRecord]:
    return [extract_from_pdf(p) for p in pdf_paths]
