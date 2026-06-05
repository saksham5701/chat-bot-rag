from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from .vector_store import FAISSVectorStore
from .chunker import Chunk

if TYPE_CHECKING:
    from .embeddings import EmbeddingsClient


class Retriever:
    """Simple retriever that uses an embeddings client and FAISS vector store.

    Responsibilities:
    - Build an index from a list of `Chunk` objects using the provided embedder.
    - Run a query and return ranked `(Chunk, score)` hits.
    """

    def __init__(self, embedder: EmbeddingsClient, store: FAISSVectorStore):
        self.embedder = embedder
        self.store = store

    def build_index(self, chunks: List[Chunk]) -> None:
        """Encode chunk texts and add them to the FAISS store."""
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        self.store.add(vectors, chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return ranked list of `(Chunk, score)` for the query."""
        q_vec = self.embedder.embed_query(query)
        results = self.store.search(q_vec, top_k=top_k)
        return results[0] if results else []
