import types
import numpy as np


def test_sentence_transformer_embedding_provider(monkeypatch):
    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            return np.array([[0.3, 0.4] for _ in texts], dtype=np.float32)

    fake_sentence_transformers = types.SimpleNamespace(SentenceTransformer=FakeModel)

    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    from app.embeddings import EmbeddingsClient

    client = EmbeddingsClient(model_name="BAAI/bge-small-en-v1.5")
    vectors = client.embed(["a"])

    assert isinstance(vectors, list)
    assert isinstance(vectors[0], np.ndarray)
    assert vectors[0].dtype == np.float32
    assert vectors[0].shape == (2,)
    assert np.allclose(vectors[0], np.array([0.3, 0.4], dtype=np.float32))
