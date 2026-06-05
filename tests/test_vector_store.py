from pathlib import Path

import numpy as np

from app.chunker import Chunk
from app.vector_store import FAISSVectorStore


def test_faiss_vector_store_add_and_search():
    chunks = [
        Chunk(id="doc:0", text="first", source="source", page_number=1),
        Chunk(id="doc:1", text="second", source="source", page_number=1),
        Chunk(id="doc:2", text="third", source="source", page_number=1),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = FAISSVectorStore(dimension=2)
    store.add(vectors, chunks)

    query = np.array([1.0, 1.0], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0][0].id == "doc:2"
    assert results[0][1][0].id in {"doc:0", "doc:1"}
    assert results[0][0][1] >= results[0][1][1]


def test_faiss_vector_store_save_and_load(tmp_path):
    chunks = [Chunk(id="doc:1", text="one", source="source", page_number=1)]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    store = FAISSVectorStore(dimension=2)
    store.add(vectors, chunks)

    base_path = tmp_path / "vector_store"
    store.save(base_path)

    loaded = FAISSVectorStore.load(base_path)
    assert loaded.dimension == 2
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0].id == "doc:1"

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = loaded.search(query, top_k=1)
    assert results[0][0][0].id == "doc:1"
    assert results[0][0][1] > 0.99
