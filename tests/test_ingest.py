from pathlib import Path
from app.ingest import normalize_text, load_documents


def test_normalize_text_collapses_whitespace():
    s = "This  is\n\na\ttest.  "
    assert normalize_text(s) == "This is a test."


def test_load_documents_empty_dir(tmp_path: Path):
    # Empty directory should yield no documents and not raise
    docs = list(load_documents(tmp_path))
    assert docs == []
