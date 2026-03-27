from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import get_settings
from src.llm.provider import OllamaProvider
from src.pipeline.orchestrator import run_pipeline


st.set_page_config(page_title="Adversarial RAG + Self-Correction", layout="wide")
st.title("🛡️ Adversarial RAG & Self Correction Pipeline")
st.caption("Ask questions. Get grounded answers with citations.")


@st.cache_resource
def get_provider(base_url: str, model: str, embedding_model: str) -> OllamaProvider:
    return OllamaProvider(
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
    )


def ask_question(
    provider: OllamaProvider,
    settings,
    question: str,
    top_k: int,
    do_reindex: bool,
    do_self_correct: bool,
    docs_dir: Path,
    adversarial_mode: bool,
) -> dict[str, Any]:
    PIPELINE_TIMEOUT_S = 35
    FAST_RETRY_TIMEOUT_S = 15

    def emergency_answer(q: str) -> str:
        _ = q
        return "I couldn’t complete retrieval in time. Try again, or enable a one-time reindex."

    def _run_once(*, k: int, reindex: bool, self_correct: bool):
        return run_pipeline(
            provider=provider,
            question=question,
            docs_dir=docs_dir,
            index_dir=settings.index_dir,
            top_k=k,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            reindex=reindex,
            enable_self_correction=self_correct,
        )

    t0 = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_once, k=top_k, reindex=do_reindex, self_correct=do_self_correct)
            result = fut.result(timeout=PIPELINE_TIMEOUT_S)

        total_time = time.perf_counter() - t0
        final_answer = (result.final_answer or result.draft_answer or "No answer generated.").strip()
        return {
            "question": question,
            "contexts": result.contexts,
            "draft_answer": result.draft_answer,
            "final_answer": final_answer,
            "latency_s": total_time,
        }
    except concurrent.futures.TimeoutError:
        pass
    except Exception as exc:
        return {
            "question": question,
            "contexts": [],
            "draft_answer": "",
            "final_answer": f"⚠️ Pipeline error: {exc}",
            "latency_s": time.perf_counter() - t0,
        }

    # Retry once in fast mode when timeout happens.
    if adversarial_mode:
        # In adversarial mode, avoid direct-answer fallback that can mask poisoning behavior.
        total_time = time.perf_counter() - t0
        return {
            "question": question,
            "contexts": [],
            "draft_answer": "",
            "final_answer": "⚠️ Retrieval timed out in adversarial mode. Try one-time reindex and ask again.",
            "latency_s": total_time,
        }

    # Retry once in fast mode when timeout happens.
    retry_t0 = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_once, k=1, reindex=False, self_correct=False)
            result = fut.result(timeout=FAST_RETRY_TIMEOUT_S)

        total_time = time.perf_counter() - retry_t0
        final_answer = (result.final_answer or result.draft_answer or "No answer generated.").strip()
        if final_answer:
            final_answer += "\n\n_(Recovered via fast retry after timeout.)_"
        return {
            "question": question,
            "contexts": result.contexts,
            "draft_answer": result.draft_answer,
            "final_answer": final_answer,
            "latency_s": total_time,
        }
    except Exception:
        # Final fallback: deterministic emergency answer so user is never blocked.
        total_time = time.perf_counter() - retry_t0
        fallback_answer = emergency_answer(question)
        return {
            "question": question,
            "contexts": [],
            "draft_answer": "",
            "final_answer": (fallback_answer or "No answer generated.").strip() + "\n\n_(Fallback response: retrieval timed out.)_",
            "latency_s": total_time,
        }


def main() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Chat Settings")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=20, value=settings.top_k)
        do_reindex = st.checkbox("Rebuild index once on next message", value=False, key="reindex_once")
        do_self_correct = st.checkbox("Enable self-correction (slower)", value=False)
        adversarial_mode = st.checkbox("Adversarial mode (poisoned docs only)", value=False)
        clear_chat = st.button("Clear Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if clear_chat:
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    provider = get_provider(
        base_url=settings.ollama_base_url,
        model=settings.victim_model,
        embedding_model=settings.embedding_model,
    )

    docs_dir = settings.docs_dir
    effective_reindex = do_reindex
    effective_self_correct = do_self_correct
    if adversarial_mode:
        docs_dir = settings.docs_dir / "adversarial"
        effective_reindex = True
        effective_self_correct = False

    user_question = st.chat_input("Ask a question...")
    if user_question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "question": user_question,
            }
        )

        with st.spinner("Running adversarial RAG pipeline..."):
            try:
                item = ask_question(
                    provider=provider,
                    settings=settings,
                    question=user_question,
                    top_k=top_k,
                    do_reindex=effective_reindex,
                    do_self_correct=effective_self_correct,
                    docs_dir=docs_dir,
                    adversarial_mode=adversarial_mode,
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        **item,
                    }
                )
            except Exception as exc:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "question": user_question,
                        "contexts": [],
                        "draft_answer": "",
                        "final_answer": f"⚠️ Pipeline failed: {exc}",
                        "latency_s": 0.0,
                    }
                )
                st.error(f"Pipeline failed: {exc}")
            finally:
                if do_reindex:
                    st.session_state["reindex_once"] = False

    st.markdown("## Chat")
    if not st.session_state.chat_history:
        st.info("Start chatting below.")
    else:
        for item in st.session_state.chat_history:
            if item.get("role") == "user":
                with st.chat_message("user"):
                    st.write(item["question"])
                continue

            with st.chat_message("assistant"):
                st.write(item["final_answer"])
                with st.expander("Details"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Retrieved Chunks", len(item["contexts"]))
                    with c2:
                        st.metric("Total Time (s)", f"{item['latency_s']:.2f}")

                    st.markdown("**Draft Answer**")
                    st.write(item["draft_answer"])

                    st.markdown("**Retrieved Contexts**")
                    for i, ctx in enumerate(item["contexts"], start=1):
                        st.markdown(
                            f"**{i}. {ctx['source']} | chunk={ctx['chunk_id']} | distance={ctx['distance']:.4f}**"
                        )
                        st.write(ctx["text"])


if __name__ == "__main__":
    main()
