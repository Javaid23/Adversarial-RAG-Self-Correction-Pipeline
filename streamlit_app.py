from __future__ import annotations

import time

import streamlit as st

from src.config import get_settings
from src.llm.provider import OllamaProvider
from src.pipeline.ingest import build_index
from src.pipeline.orchestrator import run_pipeline


st.set_page_config(page_title="Adversarial RAG + Self-Correction", layout="wide")
st.title("🛡️ Adversarial RAG + Self-Correction")
st.caption("Streamlit frontend for FEVER/SQuAD/Hotpot evaluation pipeline")


@st.cache_resource
def get_provider(base_url: str, model: str, embedding_model: str) -> OllamaProvider:
    return OllamaProvider(
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
    )


def main() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Settings")
        question = st.text_area(
            "Question",
            value="What are key adversarial risks in RAG systems?",
            height=120,
        )
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=20, value=settings.top_k)
        do_reindex = st.checkbox("Rebuild index before run", value=False)
        run_btn = st.button("Run Pipeline", type="primary")

    st.markdown("### Workflow")
    st.write("Retrieve context → Draft answer → Self-corrected final answer")

    provider = get_provider(
        base_url=settings.ollama_base_url,
        model=settings.victim_model,
        embedding_model=settings.embedding_model,
    )

    if run_btn:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Running adversarial RAG pipeline..."):
            t0 = time.perf_counter()

            if do_reindex:
                chunk_count = build_index(
                    provider=provider,
                    docs_dir=settings.docs_dir,
                    index_dir=settings.index_dir,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                st.success(f"Index rebuilt with {chunk_count} chunks")

            result = run_pipeline(
                provider=provider,
                question=question,
                docs_dir=settings.docs_dir,
                index_dir=settings.index_dir,
                top_k=top_k,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                reindex=False,
            )

            total_time = time.perf_counter() - t0

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Retrieved Chunks", len(result.contexts))
        with c2:
            st.metric("Total Time (s)", f"{total_time:.2f}")

        st.markdown("## Retrieved Contexts")
        for i, ctx in enumerate(result.contexts, start=1):
            with st.expander(f"{i}. {ctx['source']} | chunk={ctx['chunk_id']} | distance={ctx['distance']:.4f}"):
                st.write(ctx["text"])

        st.markdown("## Draft Answer")
        st.write(result.draft_answer)

        st.markdown("## Final Self-Corrected Answer")
        st.write(result.final_answer)


if __name__ == "__main__":
    main()
