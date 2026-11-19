"""Load Markdown, plain-text and PDF files into ``Document`` objects."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class Document:
    id: str  # stable, path-derived, e.g. "contracts/kaveri-foods"
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((p.extract_text() or "") for p in reader.pages)


def load_file(path: Path, root: Path) -> Document:
    rel = path.relative_to(root)
    doc_id = str(rel.with_suffix("")).replace("\\", "/")
    text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^#\s+(.+)$", text, re.M)
    title = m.group(1).strip() if m else path.stem.replace("-", " ").title()
    doc_type = rel.parts[0] if len(rel.parts) > 1 else "root"
    return Document(id=doc_id, title=title, text=text,
                    metadata={"source": str(rel), "doc_type": doc_type, "chars": len(text)})


def load_documents(root: Path) -> list[Document]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    return [load_file(p, root) for p in files]
