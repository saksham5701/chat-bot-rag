import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingsClient:
    """SentenceTransformer embeddings client.

    Uses a local `sentence-transformers` model and returns numpy arrays of dtype float32.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts and return list of numpy arrays (float32)."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [np.asarray(vec, dtype=np.float32) for vec in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query and return a numpy vector."""
        return self.embed([query])[0]
