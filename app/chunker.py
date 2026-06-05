from dataclasses import dataclass
from typing import List

from .ingest import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page_number: int


def split_text_recursive(text: str, chunk_size: int, separators: List[str] | None = None) -> List[str]:
    separators = separators or ["\n\n", "\n", " ", ""]
    normalized = text.strip()
    if len(normalized) <= chunk_size:
        return [normalized]

    for sep in separators:
        if sep == "":
            break

        parts = [part.strip() for part in normalized.split(sep) if part.strip()]
        if len(parts) <= 1:
            continue

        chunks: List[str] = []
        buffer = ""

        for part in parts:
            candidate = f"{buffer}{sep}{part}".strip() if buffer else part
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(part) > chunk_size:
                    chunks.extend(split_text_recursive(part, chunk_size, separators[1:]))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            chunks.append(buffer)

        if chunks and all(len(chunk) <= chunk_size for chunk in chunks):
            return chunks

    # Fallback: force split by raw length if no separator strategy worked.
    return [normalized[i : i + chunk_size].strip() for i in range(0, len(normalized), chunk_size)]


def create_overlapping_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    step = max(chunk_size - overlap, 1)
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += step

    return chunks


def chunk_document(document: Document, chunk_size: int, overlap: int) -> List[Chunk]:
    page_chunks = split_text_recursive(document.text, chunk_size)
    result: List[Chunk] = []

    for chunk_index, page_chunk in enumerate(page_chunks):
        for sub_index, chunk_text in enumerate(create_overlapping_chunks(page_chunk, chunk_size, overlap)):
            chunk_id = f"{document.source}:{document.page_number}:{chunk_index}:{sub_index}"
            result.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=document.source,
                    page_number=document.page_number,
                )
            )

    return result
