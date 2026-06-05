from app.chunker import chunk_document, create_overlapping_chunks, split_text_recursive
from app.ingest import Document


def test_split_text_recursive_preserves_paragraphs():
    text = "First paragraph.\n\nSecond paragraph is longer and should remain separate if chunk size allows."
    chunks = split_text_recursive(text, chunk_size=80)

    assert len(chunks) == 2
    assert chunks[0].startswith("First paragraph")
    assert chunks[1].startswith("Second paragraph")


def test_create_overlapping_chunks_builds_overlap():
    text = "A" * 120
    chunks = create_overlapping_chunks(text, chunk_size=50, overlap=10)

    assert len(chunks) == 3
    assert chunks[0][-10:] == chunks[1][:10]
    assert chunks[1][-10:] == chunks[2][:10]


def test_chunk_document_metadata_and_id():
    document = Document(source="data/sample.pdf", page_number=1, text="This is a test page. " * 10)
    chunks = chunk_document(document, chunk_size=50, overlap=10)

    assert all(chunk.source == document.source for chunk in chunks)
    assert all(chunk.page_number == document.page_number for chunk in chunks)
    assert all(chunk.id.startswith(document.source) for chunk in chunks)
    assert len(chunks) > 1
