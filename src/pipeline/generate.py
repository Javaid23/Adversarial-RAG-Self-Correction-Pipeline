from __future__ import annotations

from src.llm.provider import OllamaProvider


def generate_answer(
    provider: OllamaProvider,
    question: str,
    contexts: list[dict],
) -> str:
    joined_context = "\n\n".join(
        [f"[Source: {c['source']} | Chunk: {c['chunk_id']}]\n{c['text']}" for c in contexts]
    )

    system_prompt = (
        "You are a grounded RAG assistant. "
        "Answer only using the provided context. "
        "If context is insufficient, clearly say so. "
        "Always include concise citations in [source] format."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{joined_context}\n\n"
        "Return a concise, factual answer with citations."
    )

    return provider.chat(system_prompt, user_prompt)
