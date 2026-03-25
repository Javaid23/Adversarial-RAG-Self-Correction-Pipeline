from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _safe_name(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\- ]+", "", text).strip().replace(" ", "_")
    return (cleaned[:max_len] or "item").lower()


def _load_dataset_with_fallback(candidates: list[tuple[str, str | None]], split: str):
    from datasets import load_dataset

    errors: list[str] = []
    for name, config in candidates:
        try:
            if config is None:
                return load_dataset(name, split=split)
            return load_dataset(name, config, split=split)
        except Exception as exc:  # pragma: no cover - network/schema variability
            errors.append(f"{name}/{config or '-'} -> {exc}")
    raise RuntimeError("Could not load dataset with any fallback.\n" + "\n".join(errors))


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    docs_dir = root / "data" / "docs"
    eval_dir = root / "data" / "eval"
    docs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir, eval_dir


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _label_to_text(label: Any) -> str:
    if isinstance(label, str):
        return label.upper()
    mapping = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
    return mapping.get(label, str(label).upper())


def prepare_fever(root: Path, max_examples: int) -> dict[str, int]:
    docs_dir = root / "data" / "docs" / "fever"
    eval_path = root / "data" / "eval" / "fever_eval.jsonl"

    ds = _load_dataset_with_fallback(
        candidates=[("climate_fever", None)],
        split="test",
    )

    eval_rows: list[dict[str, Any]] = []
    written = 0
    for row in tqdm(ds, desc="Preparing FEVER"):
        claim = str(row.get("claim", "")).strip()
        if not claim:
            continue

        claim_id = row.get("claim_id", written)
        label_text = _label_to_text(row.get("claim_label", "NOT ENOUGH INFO"))
        evidences = row.get("evidences", []) or []
        evidence_lines: list[str] = []
        for ev in evidences[:5]:
            if not isinstance(ev, dict):
                continue
            article = str(ev.get("article", "")).strip()
            evidence = str(ev.get("evidence", "")).strip()
            ev_label = _label_to_text(ev.get("evidence_label", ""))
            if evidence:
                evidence_lines.append(f"- ({ev_label}) {article}: {evidence}")

        evidence_block = "\n".join(evidence_lines) if evidence_lines else "- None provided"
        filename = f"fever_{claim_id}_{_safe_name(claim[:40])}.txt"
        file_path = docs_dir / filename

        doc_text = (
            "[dataset] Climate-FEVER\n"
            f"[claim_id] {claim_id}\n"
            f"[label] {label_text}\n"
            f"[claim] {claim}\n"
            "[candidate_evidence]\n"
            f"{evidence_block}\n"
            "[note] Use this as adversarial contradiction evidence with inline citation.\n"
        )
        _write_text(file_path, doc_text)

        eval_rows.append(
            {
                "dataset": "fever",
                "question": f"Is this claim true based on retrieved evidence: {claim}",
                "expected_label": label_text,
                "citation_hint": str(file_path).replace("\\", "/"),
            }
        )

        written += 1
        if written >= max_examples:
            break

    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"docs": written, "eval": len(eval_rows)}


def prepare_squad_v2(root: Path, max_examples: int) -> dict[str, int]:
    docs_dir = root / "data" / "docs" / "squad_v2"
    eval_path = root / "data" / "eval" / "squad_v2_eval.jsonl"

    ds = _load_dataset_with_fallback(
        candidates=[("squad_v2", None)],
        split="train",
    )

    eval_rows: list[dict[str, Any]] = []
    written = 0
    for row in tqdm(ds, desc="Preparing SQuAD v2"):
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        if not context or not question:
            continue

        qid = row.get("id", written)
        answers = row.get("answers", {}) or {}
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        is_impossible = bool(row.get("is_impossible", False))

        filename = f"squadv2_{_safe_name(str(qid), max_len=50)}.txt"
        file_path = docs_dir / filename

        doc_text = (
            "[dataset] SQuAD_v2\n"
            f"[question_id] {qid}\n"
            f"[question] {question}\n"
            f"[is_impossible] {is_impossible}\n"
            "[context]\n"
            f"{context}\n"
        )
        _write_text(file_path, doc_text)

        eval_rows.append(
            {
                "dataset": "squad_v2",
                "question": question,
                "expected_answers": answer_texts,
                "requires_information_not_found": is_impossible,
                "citation_hint": str(file_path).replace("\\", "/"),
            }
        )

        written += 1
        if written >= max_examples:
            break

    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"docs": written, "eval": len(eval_rows)}


def prepare_hotpotqa(root: Path, max_examples: int) -> dict[str, int]:
    docs_dir = root / "data" / "docs" / "hotpotqa"
    eval_path = root / "data" / "eval" / "hotpotqa_eval.jsonl"

    ds = _load_dataset_with_fallback(
        candidates=[("hotpot_qa", "distractor")],
        split="train",
    )

    eval_rows: list[dict[str, Any]] = []
    written = 0
    for row in tqdm(ds, desc="Preparing HotpotQA"):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context = row.get("context", {}) or {}

        titles = context.get("title", []) if isinstance(context, dict) else []
        sentences = context.get("sentences", []) if isinstance(context, dict) else []

        if not question or not titles or not sentences:
            continue

        doc_parts: list[str] = ["[dataset] HotpotQA_distractor"]
        for title, sent_list in zip(titles, sentences):
            paragraph = " ".join(sent_list)
            doc_parts.append(f"[title] {title}\n{paragraph}")

        qid = row.get("id", written)
        filename = f"hotpot_{_safe_name(str(qid), max_len=50)}.txt"
        file_path = docs_dir / filename
        _write_text(file_path, "\n\n".join(doc_parts) + "\n")

        supporting_facts = row.get("supporting_facts", {}) or {}
        sup_titles = (
            supporting_facts.get("title", [])
            if isinstance(supporting_facts, dict)
            else []
        )

        eval_rows.append(
            {
                "dataset": "hotpotqa",
                "question": question,
                "expected_answer": answer,
                "supporting_titles": sup_titles,
                "citation_hint": str(file_path).replace("\\", "/"),
            }
        )

        written += 1
        if written >= max_examples:
            break

    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"docs": written, "eval": len(eval_rows)}


def prepare_beir_scifact_optional(root: Path, max_examples: int) -> dict[str, int]:
    docs_dir = root / "data" / "docs" / "beir_scifact"

    candidates = [
        ("BeIR/scifact", "corpus"),
        ("beir/scifact", "corpus"),
        ("mteb/scifact", "corpus"),
    ]

    ds = _load_dataset_with_fallback(candidates=candidates, split="train")

    written = 0
    for row in tqdm(ds, desc="Preparing BEIR (SciFact)"):
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        doc_id = row.get("_id", written)
        filename = f"beir_scifact_{_safe_name(str(doc_id), max_len=50)}.txt"
        file_path = docs_dir / filename
        _write_text(
            file_path,
            (
                "[dataset] BEIR_SciFact\n"
                f"[doc_id] {doc_id}\n"
                f"[title] {title}\n"
                "[text]\n"
                f"{text}\n"
            ),
        )

        written += 1
        if written >= max_examples:
            break

    return {"docs": written, "eval": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FEVER + SQuAD v2 + HotpotQA datasets into data/docs and data/eval"
    )
    parser.add_argument("--max-fever", type=int, default=250)
    parser.add_argument("--max-squad", type=int, default=250)
    parser.add_argument("--max-hotpot", type=int, default=120)
    parser.add_argument("--include-beir", action="store_true")
    parser.add_argument("--max-beir", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    _ensure_dirs(root)

    summary: dict[str, dict[str, int] | str] = {}

    summary["fever"] = prepare_fever(root, max_examples=args.max_fever)
    summary["squad_v2"] = prepare_squad_v2(root, max_examples=args.max_squad)
    summary["hotpotqa"] = prepare_hotpotqa(root, max_examples=args.max_hotpot)

    if args.include_beir:
        try:
            summary["beir_scifact"] = prepare_beir_scifact_optional(
                root,
                max_examples=args.max_beir,
            )
        except Exception as exc:  # pragma: no cover - optional path
            summary["beir_scifact"] = f"Skipped due to load error: {exc}"

    manifest_path = root / "data" / "eval" / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDataset preparation complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
