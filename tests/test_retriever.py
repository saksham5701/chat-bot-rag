import numpy as np

from app.chunker import Chunk
from app.retriever import Retriever
from app.vector_store import FAISSVectorStore


class FakeEmbedder:
    def embed(self, texts):
        mapping = {
            "first": [1.0, 0.0],
            "second": [0.0, 1.0],
            "third": [1.0, 1.0],
        }
        return [np.array(mapping.get(t, [0.0, 0.0]), dtype=np.float32) for t in texts]

    def embed_query(self, query: str):
        return np.array([1.0, 1.0], dtype=np.float32)


def test_retriever_build_and_query():
    chunks = [
        Chunk(id="doc:0", text="first", source="s", page_number=1),
        Chunk(id="doc:1", text="second", source="s", page_number=1),
        Chunk(id="doc:2", text="third", source="s", page_number=1),
    ]

    embedder = FakeEmbedder()
    store = FAISSVectorStore(dimension=2)
    retriever = Retriever(embedder, store)

    retriever.build_index(chunks)
    results = retriever.retrieve("anything", top_k=2)

    assert len(results) == 2
    assert results[0][0].id == "doc:2"
    assert results[1][0].id in {"doc:0", "doc:1"}
