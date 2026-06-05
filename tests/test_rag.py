import numpy as np

from app.chunker import Chunk
from app.rag import RAG
from app.retriever import Retriever


class FakeEmbedder:
    def embed(self, texts):
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

    def embed_query(self, query: str):
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        return "generated-answer"


def test_rag_generate_context_and_prompt():
    chunks = [
        Chunk(id="doc:0", text="chunk one", source="source.pdf", page_number=1),
        Chunk(id="doc:1", text="chunk two", source="source.pdf", page_number=2),
    ]

    from app.vector_store import FAISSVectorStore

    store = FAISSVectorStore(dimension=2)
    retriever = Retriever(FakeEmbedder(), store)
    retriever.build_index(chunks)

    llm = FakeLLM()
    rag = RAG(retriever, llm, max_chunks=2)

    context = rag.generate_context("query text", top_k=2)
    assert "chunk one" in context
    assert "chunk two" in context
    assert "Source: source.pdf" in context

    prompt = rag.build_prompt("query text", context)
    assert "Use the following context" in prompt
    assert "Question: query text" in prompt

    answer = rag.answer("query text", top_k=2)
    assert answer == "generated-answer"
    assert llm.last_prompt is not None
    assert "chunk one" in llm.last_prompt
    assert "chunk two" in llm.last_prompt
