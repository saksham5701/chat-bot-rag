from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from .chunker import Chunk


class FAISSVectorStore:
    """A simple FAISS-backed vector store for chunk retrieval."""

    def __init__(self, dimension: int, normalize: bool = True, index: Optional[faiss.Index] = None):
        self.dimension = dimension
        self.normalize = normalize
        self.index = index or faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Chunk] = []

    def _ensure_dimension(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vectors must be a 2D array with dimension {self.dimension}; got {vectors.shape}"
            )

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors
        vectors = vectors.astype(np.float32, copy=False)
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        """Add a batch of vectors and the corresponding chunk metadata."""
        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must match number of chunks.")

        vectors = np.asarray(vectors, dtype=np.float32)
        self._ensure_dimension(vectors)
        vectors = self._normalize(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> List[List[tuple[Chunk, float]]]:
        """Search the FAISS index and return ranked chunk results with similarity scores."""
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        self._ensure_dimension(query_vectors)
        query_vectors = self._normalize(query_vectors)

        scores, indices = self.index.search(query_vectors, top_k)
        results: List[List[tuple[Chunk, float]]] = []

        for row_scores, row_indices in zip(scores, indices):
            hits: List[tuple[Chunk, float]] = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                hits.append((self.chunks[idx], float(score)))
            results.append(hits)

        return results

    def save(self, path: Path) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".index")))
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "normalize": self.normalize,
                    "chunks": self.chunks,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "FAISSVectorStore":
        """Load a persisted FAISS index and its chunk metadata."""
        path = Path(path)
        index = faiss.read_index(str(path.with_suffix(".index")))
        with open(path.with_suffix(".pkl"), "rb") as f:
            data = pickle.load(f)

        store = cls(dimension=data["dimension"], normalize=data["normalize"], index=index)
        store.chunks = data["chunks"]
        return store
