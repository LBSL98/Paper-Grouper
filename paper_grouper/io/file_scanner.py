from pathlib import Path
from typing import List


def list_pdfs(folder: str) -> List[str]:
    p = Path(folder)
    return [
        str(f.resolve())
        for f in p.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]
