from src.eval.metrics import citation_coverage, contains_uncertainty_when_needed


def test_citation_coverage_non_zero():
    answer = "RAG helps reduce hallucinations [doc1]."
    assert citation_coverage(answer) > 0


def test_uncertainty_detector():
    answer = "The evidence is insufficient to determine the final cause."
    assert contains_uncertainty_when_needed(answer)
