from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma

from src.llm.provider import OllamaProvider


def retrieve_contexts(
    provider: OllamaProvider,
    question: str,
    index_dir: Path,
    top_k: int,
) -> list[dict]:
    # Ensure path is a Path object
    index_dir = Path(index_dir) if not isinstance(index_dir, Path) else index_dir
    
    persist_dir = index_dir / "chroma"

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Chroma index not found at '{persist_dir}'. Rebuild the index (or run pipeline once to auto-build)."
        )

    vectorstore = Chroma(
        collection_name="rag_chunks",
        persist_directory=str(persist_dir),
        embedding_function=provider.get_embedding_function(),
    )
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)

    results: list[dict] = []
    for doc, score in docs_with_scores:
        metadata = doc.metadata or {}
        results.append(
            {
                "chunk_id": int(metadata.get("id", -1)),
                "source": str(metadata.get("source", "unknown")),
                "text": doc.page_content,
                "distance": float(score),
            }
        )
    return results
