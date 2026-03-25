from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline.generate import generate_answer
from src.pipeline.ingest import build_index
from src.pipeline.retrieve import retrieve_contexts
from src.pipeline.self_correct import critique_and_revise


@dataclass
class PipelineResult:
    contexts: list[dict]
    draft_answer: str
    final_answer: str


def run_pipeline(
    provider,
    question: str,
    docs_dir,
    index_dir,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    reindex: bool = False,
) -> PipelineResult:
    persist_dir = Path(index_dir) / "chroma"

    if reindex or not persist_dir.exists():
        build_index(
            provider=provider,
            docs_dir=docs_dir,
            index_dir=index_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    contexts = retrieve_contexts(
        provider=provider,
        question=question,
        index_dir=index_dir,
        top_k=top_k,
    )
    draft_answer = generate_answer(provider=provider, question=question, contexts=contexts)
    final_answer = critique_and_revise(
        provider=provider,
        question=question,
        draft_answer=draft_answer,
        contexts=contexts,
    )

    return PipelineResult(
        contexts=contexts,
        draft_answer=draft_answer,
        final_answer=final_answer,
    )
