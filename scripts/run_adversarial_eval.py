#!/usr/bin/env python3
"""
Run the adversarial robustness evaluation.
Generates a report showing the "Success Rate of Failure" - how often the LLM
was tricked by poisoned data, and how well self-correction and verification
techniques mitigate this.

Usage:
    python scripts/run_adversarial_eval.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.adversarial_benchmark import (
    load_adversarial_eval,
    evaluate_adversarial_robustness,
    compute_metrics,
    generate_report,
)
from src.config import get_settings
from src.llm.provider import OllamaProvider


def main():
    print("=" * 100)
    print("ADVERSARIAL RAG & SELF-CORRECTION PIPELINE")
    print("Deliverable 1: Adversarial Robustness Evaluation")
    print("=" * 100)
    print()
    
    try:
        settings = get_settings()
        provider = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.victim_model,
            embedding_model=settings.embedding_model,
        )
        
        print(f"Configuration:")
        print(f"  Model: {settings.victim_model}")
        print(f"  Embedding: {settings.embedding_model}")
        print(f"  Base URL: {settings.ollama_base_url}")
        print()
        
        # Load adversarial evaluation set
        eval_file = Path("data/eval/adversarial_poison_eval.jsonl")
        if not eval_file.exists():
            print(f"❌ Error: Evaluation file not found at {eval_file}")
            print("Please run: python scripts/prepare_datasets.py")
            return False
        
        eval_items = load_adversarial_eval(eval_file)
        print(f"Loaded {len(eval_items)} adversarial test cases")
        print()
        
        # Run evaluation
        print("Starting adversarial robustness evaluation...")
        print("This will test the pipeline against poisoned documents to measure:")
        print("  1. How often the LLM is 'tricked' by false information (Baseline)")
        print("  2. How self-correction improves robustness")
        print("  3. How CoVe verification protects against hallucinations")
        print("  4. How Self-RAG critique filters bad contexts")
        print()
        
        results = evaluate_adversarial_robustness(
            provider=provider,
            docs_dir=settings.docs_dir / "adversarial",
            index_dir=settings.index_dir / "adversarial_eval",
            eval_items=eval_items,
            enable_self_correction=True,
            enable_cove=True,
            enable_self_rag=True,
        )
        
        # Compute metrics
        metrics = compute_metrics(results)
        
        # Generate report
        report_file = Path("results/deliverable_1_adversarial_robustness_report.txt")
        generate_report(results, metrics, report_file)
        
        # Save detailed results as JSON
        import json
        json_file = Path("results/adversarial_eval_detailed.json")
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        
        print()
        print("=" * 100)
        print("✅ EVALUATION COMPLETE")
        print("=" * 100)
        print(f"Report saved to: {report_file}")
        print(f"Detailed results saved to: {json_file}")
        print()
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
