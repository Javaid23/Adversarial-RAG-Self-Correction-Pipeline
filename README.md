# Adversarial RAG + Self-Correction Pipeline

A production-ready Python project for building **adversarially robust Retrieval-Augmented Generation (RAG)** systems with **self-correction and verification layers**.

## 🎯 Project Objectives

This project implements the complete adversarial RAG pipeline as specified in the assignment:

### Phase One: Adversarial Setup ✓
- **Poisoned Knowledge Base**: Custom dataset with contradictory facts, temporal lapses, and distractor noise
- **Stress Testing**: 8 hand-crafted adversarial queries designed to trick the LLM
- **Deliverable 1**: Automated "Success Rate of Failure" report showing how often the baseline RAG produces confident hallucinations

### Phase Two: The Mitigation ✓
- **Chain-of-Verification (CoVe)**: Model generates verification questions, answers them independently, and revises based on findings
- **Self-RAG**: Specialized tokens (IsRel, IsSup, IsUseful) for critiquing retrieval quality and relevance
- **Self-Correction**: Critique-and-revise loop for grounding answers in provided context

## What's Included

- ✅ **Interactive Streamlit App**: Test the adversarial RAG system in real-time
- ✅ **Automated Evaluation Suite**: Generate comprehensive robustness reports
- ✅ **Multiple Datasets**: FEVER, SQuAD v2, HotpotQA, + custom adversarial poison set
- ✅ **Citation Tracking**: Inline citations with "Information not found" fallback
- ✅ **Latency Measurements**: Time to first token and end-to-end latency tracking
- ✅ **No Hardcoding**: All verification techniques are generalizable and model-agnostic

## Tech Stack

- **Framework**: LangChain (for orchestration)
- **Victim Model**: Ollama-hosted open-weights LLM (default: `llama3:8b`)
- **Vector Database**: ChromaDB (fast, embedded, no external dependencies)
- **Frontend**: Streamlit (interactive testing)
- **Evaluation**: Custom Python scripts + Jupyter notebooks

## Architecture

```text
src/
  config.py                 # environment + runtime settings
  main.py                   # CLI entrypoint
  pipeline/
    ingest.py               # load/chunk/index documents
    retrieve.py             # vector similarity search
    generate.py             # first-pass answer generation with citations
    self_correct.py         # critique-and-revise mitigation
    cove.py                 # Chain-of-Verification implementation
    self_rag.py             # Self-RAG retrieval quality critique
    orchestrator.py         # orchestrate full pipeline
  llm/
    provider.py             # Ollama + LangChain wrapper
  eval/
    metrics.py              # citation coverage, uncertainty detection
    adversarial_benchmark.py # Deliverable 1: robustness evaluation
    benchmark.py            # general benchmarking

notebooks/
  adversarial_evaluation.ipynb  # Interactive evaluation notebook

scripts/
  prepare_datasets.py       # Prepare FEVER/SQuAD/HotpotQA datasets
  run_adversarial_eval.py   # Run Deliverable 1 report generation
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Prepare Datasets

```bash
python scripts/prepare_datasets.py --max-fever 250 --max-squad 250 --max-hotpot 120
```

### 3. Start Ollama (in a separate terminal)

```bash
ollama serve
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### 4. Run Interactive Demo

```bash
# Option A: Streamlit interactive interface (recommended)
streamlit run streamlit_app.py

# Option B: CLI interface
python -m src.main --question "What is the capital of Germany?"
```

### 5. Generate Deliverable 1 Report

```bash
# Run automated adversarial robustness evaluation
python scripts/run_adversarial_eval.py

# Results saved to:
#   - results/deliverable_1_adversarial_robustness_report.txt
#   - results/adversarial_eval_detailed.json
#   - results/adversarial_robustness_metrics.png
#   - results/latency_vs_accuracy.png
```

### 6. Interactive Evaluation (Jupyter)

```bash
jupyter notebook notebooks/adversarial_evaluation.ipynb
```

## Dataset Strategy

- **FEVER**: Fact verification with contradiction handling (SUPPORTS/REFUTES/NOT ENOUGH INFO)
- **SQuAD v2**: Unanswerable questions and "Information not found" behavior
- **HotpotQA**: Multi-hop reasoning with distractor contexts
- **Adversarial Poison** (Custom): Hand-crafted false claims designed to trick the LLM
  - Examples: False capital names, wrong historical dates, fake scientific claims
  - Tests: How well RAG resists confident hallucinations

Manifest: `data/eval/dataset_manifest.json`

## Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434     # Ollama server URL
VICTIM_MODEL=llama3:8b                     # Model to test ("trick")
EMBEDDING_MODEL=nomic-embed-text           # Embedding model for vectors
VECTOR_DB=chroma                           # Vector database type
TOP_K=5                                    # Number of contexts to retrieve
CHUNK_SIZE=800                             # Document chunk size
CHUNK_OVERLAP=120                          # Overlap between chunks
```

## Streamlit Interface

The Streamlit app provides:
- **Real-time Q&A** against poisoned/adversarial dataset
- **Toggle Mitigation Strategies**:
  - Self-Correction (critique-and-revise)
  - CoVe (chain-of-verification)
  - Self-RAG (retrieval quality critique)
- **Performance Metrics**: Latency, number of retrieved contexts, draft vs. final answer
- **Context Inspection**: View exact documents retrieved for each query

## Deliverable 1: Adversarial Robustness Report

Generates three key metrics:

1. **Success Rate of Failure (%) - Baseline**: How often the poisoned RAG confidently produces hallucinations
2. **Effectiveness of Mitigations**: % improvement when using CoVe, Self-RAG, and Self-Correction
3. **Latency vs. Accuracy Tradeoffs**: Shows the performance cost of each mitigation strategy

Example output:
```
SUMMARY METRICS
================================================================
Baseline:
  Total Adversarial Queries: 8
  Times Model Was Tricked: 6
  ✗ Success Rate of Failure: 75.0%
  ✓ Resistance Rate: 25.0%
  Average Latency: 4.32s

With Self-Correction:
  ✗ Success Rate of Failure: 37.5%
  ✓ Resistance Rate: 62.5%
  Average Latency: 8.15s
  Improvement: 37.5%

With CoVe:
  ✗ Success Rate of Failure: 25.0%
  ✓ Resistance Rate: 75.0%
  Average Latency: 12.45s
  Improvement: 50.0%
```

## Key Design Decisions

### 1. No Hardcoding
All evaluation metrics are computed at runtime based on actual LLM outputs. No answer keys are hardcoded into the verification pipeline.

### 2. Citation Requirement
Every answer must include inline citations in `[source]` format or explicitly state "Information not found" if evidence is missing.

### 3. Latency Awareness
The system tracks and reports end-to-end latency for each strategy. Users can observe the accuracy vs. speed tradeoff.

### 4. Generalizable Mitigation
Both CoVe and Self-RAG work with any language model via the LangChain provider interface. No model-specific tuning required.

## Performance Expectations

| Strategy | Accuracy | Latency | Best For |
|----------|----------|---------|----------|
| Baseline RAG | 25-50% | 3-5s | Speed (not recommended for adversarial contexts) |
| Self-Correction | 50-70% | 7-10s | Moderate accuracy with reasonable latency |
| CoVe | 70-85% | 12-15s | High accuracy when speed permits |
| Self-RAG | 60-75% | 8-12s | Filtering bad contexts, multi-hop reasoning |

*Note: Accuracy varies based on poisoned data quality and LLM reasoning capability.*

## Next Recommended Upgrades

1. **Hybrid Retrieval**: BM25 + dense vector search
2. **Cross-Encoder Reranking**: Improved context ranking
3. **Query Rewriting**: Reformulate queries before second retrieval pass
4. **Prompt Injection Guardrails**: Detect adversarial input patterns
5. **Regression Tests**: Automated test suite for pipeline robustness
6. **Semantic Similarity Thresholds**: Reject low-confidence retrievals
7. **Multi-Model Voting**: Query multiple LLMs and combine answers

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/status

# Pull models if not available
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### Out of Memory
```bash
# Run Ollama with reduced memory
OLLAMA_NUM_PARALLEL=1 ollama serve
```

### Streamlit Issues
```bash
# Clear cache and restart
rm -rf ~/.streamlit
streamlit run streamlit_app.py --logger.level=debug
```

## Citation

If you use this pipeline in your research or projects, please reference:

```bibtex
@misc{adversarial_rag_2024,
  title={Adversarial RAG & Self-Correction Pipeline},
  author={Your Name},
  year={2024},
  note={Stress-testing RAG systems against poisoned data with CoVe and Self-RAG mitigations}
}
```

## License

MIT License - See LICENSE file for details

