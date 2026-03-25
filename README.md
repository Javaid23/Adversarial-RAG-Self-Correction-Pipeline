# Adversarial RAG + Self-Correction Pipeline

A modular Python project for building **adversarially robust Retrieval-Augmented Generation (RAG)** systems with a **self-correction loop**.

## What this starter includes

- Document ingestion and chunking
- Vector indexing with ChromaDB
- Dense retrieval + optional reranking hook
- Answer generation with citations
- Self-correction pass (faithfulness / grounding checks)
- Minimal CLI runner for iterative experiments

## Tech stack alignment (per assignment)

- **Framework:** LangChain
- **Victim model:** Ollama-hosted open-weights LLM (default `llama3:8b`)
- **Vector DB:** ChromaDB

## Initial architecture

```text
src/
  config.py              # environment + runtime settings
  main.py                # CLI entrypoint
  pipeline/
    ingest.py            # load/chunk/index docs
    retrieve.py          # retrieve top-k contexts
    generate.py          # first-pass answer generation
    self_correct.py      # critique-and-revise logic
    orchestrator.py      # end-to-end flow controller
  llm/
    provider.py          # Ollama + LangChain wrapper
  eval/
    metrics.py           # basic offline metrics placeholders
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Fill in `.env` (or copy from `.env.example`).
4. Prepare datasets (no manual poisoned-KB authoring required):
  - `python scripts/prepare_datasets.py --max-fever 250 --max-squad 250 --max-hotpot 120`
  - Optional retrieval stress corpus: add `--include-beir`
4. Run the pipeline:
   - `python -m src.main --question "What are the main risks in adversarial RAG?"`
5. Run the Streamlit frontend (recommended for demos):
  - `streamlit run streamlit_app.py`

## Dataset strategy used in this repo

- **FEVER** → contradiction handling (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`)
- **SQuAD v2** → unanswerable behavior (`Information not found` style)
- **HotpotQA (distractor)** → multi-hop + distracting context
- **BEIR (optional)** → retrieval stress testing corpus

The prep script writes:

- Retrieval documents to `data/docs/`
- Evaluation sets to `data/eval/`
- A summary manifest to `data/eval/dataset_manifest.json`

## Environment variables

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `VICTIM_MODEL` (default: `llama3:8b`)
- `EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `VECTOR_DB` (default: `chroma`)
- `TOP_K` (default: `5`)
- `CHUNK_SIZE` (default: `800`)
- `CHUNK_OVERLAP` (default: `120`)

## Next recommended upgrades

- Hybrid retrieval (BM25 + dense)
- Cross-encoder reranking
- Query rewrite before second retrieval
- Guardrails for prompt injection in retrieved chunks
- Structured evaluation set + regression tests
