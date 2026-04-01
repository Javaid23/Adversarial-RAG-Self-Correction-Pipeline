from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline.generate import generate_answer
from src.pipeline.ingest import build_index
from src.pipeline.retrieve import retrieve_contexts
from src.pipeline.self_correct import critique_and_revise
from src.pipeline.cove import cove_verify
from src.pipeline.self_rag import self_rag_critique


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
    enable_self_correction: bool = True,
    enable_cove: bool = False,
    enable_self_rag: bool = False,
) -> PipelineResult:
    # Ensure paths are Path objects
    docs_dir = Path(docs_dir) if not isinstance(docs_dir, Path) else docs_dir
    index_dir = Path(index_dir) if not isinstance(index_dir, Path) else index_dir
    
    persist_dir = index_dir / "chroma"

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
    if enable_self_rag:
        contexts, _ = self_rag_critique(
            provider=provider,
            question=question,
            contexts=contexts,
        )
    draft_answer = generate_answer(provider=provider, question=question, contexts=contexts)
    final_answer = draft_answer
    if enable_cove:
        final_answer = cove_verify(
            provider=provider,
            question=question,
            draft_answer=final_answer,
            contexts=contexts,
        )
    if enable_self_correction:
        final_answer = critique_and_revise(
            provider=provider,
            question=question,
            draft_answer=final_answer,
            contexts=contexts,
        )

    return PipelineResult(
        contexts=contexts,
        draft_answer=draft_answer,
        final_answer=final_answer,
    )
