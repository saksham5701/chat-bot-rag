from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass
class Document:
    source: str
    page_number: int
    text: str


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove redundant line breaks."""
    if text is None:
        return ""
    return " ".join(text.split())


def extract_pdf_pages(pdf_path: Path) -> Iterable[Document]:
    """
    Extract pages from a PDF and yield Document objects.
    Raises FileNotFoundError if the file does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        normalized = normalize_text(page_text)
        if normalized:
            yield Document(source=str(pdf_path), page_number=page_number, text=normalized)


def load_documents(data_dir: Path) -> Iterable[Document]:
    """
    Walk `data_dir` recursively and yield Document objects for each PDF page found.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return

    for pdf_path in sorted(data_dir.rglob("*.pdf")):
        try:
            yield from extract_pdf_pages(pdf_path)
        except Exception:
            # In production we'd log the error; keep ingestion resilient here.
            continue
