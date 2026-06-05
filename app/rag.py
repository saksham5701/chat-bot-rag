from typing import List

from .llm import LLMClient
from .retriever import Retriever


class RAG:
    """Retrieval-augmented generation pipeline."""

    def __init__(self, retriever: Retriever, llm: LLMClient, max_chunks: int = 3):
        self.retriever = retriever
        self.llm = llm
        self.max_chunks = max_chunks

    def generate_context(self, query: str, top_k: int = 5) -> str:
        hits = self.retriever.retrieve(query, top_k=top_k)
        if not hits:
            return ""

        selected = hits[: self.max_chunks]
        context_entries: List[str] = []
        for chunk, score in selected:
            context_entries.append(
                f"Source: {chunk.source}, page: {chunk.page_number}, score: {score:.4f}\n{chunk.text}"
            )

        return "\n\n".join(context_entries)

    def build_prompt(self, query: str, context: str) -> str:
        if not context:
            return f"Answer the user question if possible. If you do not know, say you do not have enough information.\n\nQuestion: {query}\nAnswer:"

        return (
            "Use the following context to answer the question. "
            "If the answer is not contained in the context, say you do not have enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

    def answer(self, query: str, top_k: int = 5) -> str:
        context = self.generate_context(query, top_k=top_k)
        prompt = self.build_prompt(query, context)
        return self.llm.generate(prompt)
