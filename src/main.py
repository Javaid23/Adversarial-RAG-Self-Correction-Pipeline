from __future__ import annotations

import argparse

from src.config import get_settings
from src.llm.provider import OllamaProvider
from src.pipeline.orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial RAG + Self-Correction Pipeline"
    )
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the vector index from data/docs before retrieval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.victim_model,
        embedding_model=settings.embedding_model,
    )

    result = run_pipeline(
        provider=provider,
        question=args.question,
        docs_dir=settings.docs_dir,
        index_dir=settings.index_dir,
        top_k=settings.top_k,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reindex=args.reindex,
    )

    print("\n=== Retrieved Contexts ===")
    for i, c in enumerate(result.contexts, start=1):
        preview = c["text"][:220].replace("\n", " ")
        print(f"{i}. {c['source']} (chunk={c['chunk_id']}, distance={c['distance']:.4f})")
        print(f"   {preview}...")

    print("\n=== Draft Answer ===")
    print(result.draft_answer)

    print("\n=== Final Self-Corrected Answer ===")
    print(result.final_answer)


if __name__ == "__main__":
    main()
