from __future__ import annotations

from src.llm.provider import OllamaProvider


def _format_contexts(contexts: list[dict]) -> str:
    return "\n\n".join(
        [f"[Source: {c['source']} | Chunk: {c['chunk_id']}]\n{c['text']}" for c in contexts]
    )


def generate_verification_questions(
    provider: OllamaProvider,
    question: str,
    draft_answer: str,
) -> list[str]:
    system_prompt = (
        "You are a verification planner. Generate short, focused fact-check questions "
        "that would validate the draft answer. Do not answer them."
    )
    user_prompt = (
        f"Original Question:\n{question}\n\n"
        f"Draft Answer:\n{draft_answer}\n\n"
        "Return 3-5 verification questions, one per line, no numbering."
    )
    raw = provider.chat(system_prompt, user_prompt).strip()
    questions = [q.strip(" -\t") for q in raw.splitlines() if q.strip()]
    return questions[:5]


def answer_verification_questions(
    provider: OllamaProvider,
    questions: list[str],
    contexts: list[dict],
) -> str:
    joined_context = _format_contexts(contexts)
    system_prompt = (
        "You are a strict verifier. Answer each verification question using ONLY the provided context. "
        "If the context is insufficient, respond with 'Information not found.'"
    )
    user_prompt = (
        "Verification Questions:\n"
        + "\n".join(questions)
        + "\n\nRetrieved Context:\n"
        + joined_context
        + "\n\nProvide concise answers in the same order, one per line."
    )
    return provider.chat(system_prompt, user_prompt)


def revise_answer_with_verification(
    provider: OllamaProvider,
    question: str,
    draft_answer: str,
    verification_questions: list[str],
    verification_answers: str,
    contexts: list[dict],
) -> str:
    joined_context = _format_contexts(contexts)
    system_prompt = (
        "You are a careful verifier. Revise the answer to ensure every claim is supported by context and "
        "consistent with the verification answers. Include inline citations. If evidence is missing, say so."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Draft Answer:\n{draft_answer}\n\n"
        "Verification Questions:\n"
        + "\n".join(verification_questions)
        + "\n\nVerification Answers:\n"
        + verification_answers
        + "\n\nRetrieved Context:\n"
        + joined_context
        + "\n\nReturn a corrected final answer with citations."
    )
    return provider.chat(system_prompt, user_prompt)


def cove_verify(
    provider: OllamaProvider,
    question: str,
    draft_answer: str,
    contexts: list[dict],
) -> str:
    questions = generate_verification_questions(provider, question, draft_answer)
    if not questions:
        return draft_answer
    answers = answer_verification_questions(provider, questions, contexts)
    return revise_answer_with_verification(
        provider=provider,
        question=question,
        draft_answer=draft_answer,
        verification_questions=questions,
        verification_answers=answers,
        contexts=contexts,
    )
