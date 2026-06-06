from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

from .chunker import Chunk, chunk_document
from .embeddings import EmbeddingsClient
from .ingest import load_documents
from .llm import LLMClient
from .rag import RAG
from .retriever import Retriever
from .vector_store import FAISSVectorStore


def build_chunks(data_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in load_documents(data_dir):
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks


def format_context(hits: Iterable[Tuple[Chunk, float]], max_chunks: int) -> str:
    context_entries: List[str] = []
    for chunk, score in list(hits)[:max_chunks]:
        context_entries.append(
            f"Source: {chunk.source}, page: {chunk.page_number}, score: {score:.4f}\n{chunk.text}"
        )
    return "\n\n".join(context_entries)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question against PDFs stored in a data folder.")
    parser.add_argument("question", help="Question to ask about the indexed PDFs.")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF files. Default: data")
    parser.add_argument("--chunk-size", type=int, default=800, help="Maximum chunk size. Default: 800")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap. Default: 100")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve. Default: 5")
    parser.add_argument("--max-chunks", type=int, default=3, help="Chunks sent to the LLM. Default: 3")
    return parser


def main() -> None:
    load_dotenv()
    args = create_parser().parse_args()

    data_dir = Path(args.data_dir)
    chunks = build_chunks(data_dir=data_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise SystemExit(f"No PDF text chunks found in {data_dir}. Add a readable PDF and try again.")

    print(f"Loaded {len(chunks)} chunks from PDFs in {data_dir}.")
    print("Loading embedding model and building FAISS index...")

    embedder = EmbeddingsClient()
    dimension = len(embedder.embed_query("embedding dimension probe"))
    store = FAISSVectorStore(dimension=dimension)
    retriever = Retriever(embedder=embedder, store=store)
    retriever.build_index(chunks)

    llm = LLMClient()
    rag = RAG(retriever=retriever, llm=llm, max_chunks=args.max_chunks)

    hits = retriever.retrieve(args.question, top_k=args.top_k)
    context = format_context(hits, max_chunks=args.max_chunks)
    prompt = rag.build_prompt(args.question, context)
    answer = llm.generate(prompt)

    print("\nAnswer")
    print("------")
    print(answer)

    print("\nSources")
    print("-------")
    for chunk, score in hits[: args.max_chunks]:
        print(f"- {chunk.source}, page {chunk.page_number}, score {score:.4f}")


if __name__ == "__main__":
    main()
