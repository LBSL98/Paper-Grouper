# paper_grouper

Literature clustering & organization tool.

## What it does (v0.1)

1. Scans a folder full of PDF articles.
2. Extracts title / abstract / keywords.
3. Builds semantic embeddings.
4. Creates a k-NN similarity graph and runs community detection (Louvain/Leiden).
5. Auto-tunes parameters (k, resolution, etc.) in parallel to get the best clustering.
6. Groups the PDFs into new folders by cluster, optionally renaming files using the paper title.
7. Generates reports (JSON/TXT) and a graph visualization highlighting central / key papers.

## Architecture

- `ui/` PySide6 GUI (no CLI needed for normal use)
- `core/` logic for metadata extraction, embeddings, graph build, clustering, autotune
- `io/` filesystem ops (scan input, copy/rename, write reports, render graph images)
- `app_controller.py` glue between GUI and core/io
- `app_entry.py` GUI entrypoint

Everything in `core/` is pure logic (no GUI, no filesystem side effects), so it can be tested and reused headless.

## Next steps
- Implement PDF metadata extraction
- Hook PySide6 widgets to controller
- Add .bib export per cluster (future)
