from __future__ import annotations

import re

from src.llm.provider import OllamaProvider


def _format_contexts(contexts: list[dict]) -> str:
    lines = []
    for idx, ctx in enumerate(contexts, start=1):
        lines.append(
            f"[{idx}] Source: {ctx['source']} | Chunk: {ctx['chunk_id']}\n{ctx['text']}"
        )
    return "\n\n".join(lines)


def self_rag_critique(
    provider: OllamaProvider,
    question: str,
    contexts: list[dict],
) -> tuple[list[dict], str]:
    if not contexts:
        return contexts, "No contexts provided."

    joined_context = _format_contexts(contexts)

    system_prompt = (
        "You are a Self-RAG critic. For each context, judge whether it is relevant, "
        "supports answering the question, and is useful. Respond with tokens: IsRel, "
        "IsSup, IsUseful using Yes/No."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Contexts:\n{joined_context}\n\n"
        "Return one line per context in this exact format:\n"
        "[index] IsRel=<Yes|No> IsSup=<Yes|No> IsUseful=<Yes|No> Rationale=<short reason>"
    )

    raw = provider.chat(system_prompt, user_prompt).strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    verdicts: dict[int, dict[str, str]] = {}
    pattern = re.compile(
        r"\[(\d+)\]\s+IsRel=(Yes|No)\s+IsSup=(Yes|No)\s+IsUseful=(Yes|No)\s+Rationale=(.*)",
        re.IGNORECASE,
    )

    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        idx = int(match.group(1))
        verdicts[idx] = {
            "IsRel": match.group(2).title(),
            "IsSup": match.group(3).title(),
            "IsUseful": match.group(4).title(),
            "Rationale": match.group(5).strip(),
        }

    filtered: list[dict] = []
    for idx, ctx in enumerate(contexts, start=1):
        verdict = verdicts.get(idx)
        if verdict is None:
            filtered.append(ctx)
            continue
        if verdict["IsRel"] == "Yes" and verdict["IsSup"] == "Yes":
            filtered.append(ctx)

    if not filtered:
        filtered = contexts

    critique_report = "\n".join(lines) if lines else raw
    return filtered, critique_report
