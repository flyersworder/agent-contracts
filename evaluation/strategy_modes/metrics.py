"""ROUGE metrics for evaluating summarization quality.

This module provides ROUGE score computation for comparing generated
summaries against reference summaries from CNN/DailyMail.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures:
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

Example:
    >>> metrics = compute_rouge(
    ...     hypothesis="The quick brown fox jumps.",
    ...     reference="A quick brown fox jumped over the lazy dog."
    ... )
    >>> print(f"ROUGE-L F1: {metrics.rouge_l_f1:.3f}")
"""

from dataclasses import dataclass


@dataclass
class RougeMetrics:
    """ROUGE metrics for a single evaluation.

    Attributes:
        rouge_1_precision: ROUGE-1 precision
        rouge_1_recall: ROUGE-1 recall
        rouge_1_f1: ROUGE-1 F1 score
        rouge_2_precision: ROUGE-2 precision
        rouge_2_recall: ROUGE-2 recall
        rouge_2_f1: ROUGE-2 F1 score
        rouge_l_precision: ROUGE-L precision
        rouge_l_recall: ROUGE-L recall
        rouge_l_f1: ROUGE-L F1 score (primary metric)
    """

    rouge_1_precision: float = 0.0
    rouge_1_recall: float = 0.0
    rouge_1_f1: float = 0.0
    rouge_2_precision: float = 0.0
    rouge_2_recall: float = 0.0
    rouge_2_f1: float = 0.0
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    rouge_l_f1: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rouge_1_precision": self.rouge_1_precision,
            "rouge_1_recall": self.rouge_1_recall,
            "rouge_1_f1": self.rouge_1_f1,
            "rouge_2_precision": self.rouge_2_precision,
            "rouge_2_recall": self.rouge_2_recall,
            "rouge_2_f1": self.rouge_2_f1,
            "rouge_l_precision": self.rouge_l_precision,
            "rouge_l_recall": self.rouge_l_recall,
            "rouge_l_f1": self.rouge_l_f1,
        }


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for ROUGE computation.

    Args:
        text: Input text

    Returns:
        List of lowercase tokens
    """
    import re

    # Convert to lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from token list.

    Args:
        tokens: List of tokens
        n: N-gram size

    Returns:
        List of n-gram tuples
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _compute_precision_recall_f1(
    hypothesis_count: int, reference_count: int, overlap_count: int
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from counts.

    Args:
        hypothesis_count: Number of items in hypothesis
        reference_count: Number of items in reference
        overlap_count: Number of overlapping items

    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = 0.0 if hypothesis_count == 0 else overlap_count / hypothesis_count
    recall = 0.0 if reference_count == 0 else overlap_count / reference_count
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence.

    Args:
        x: First sequence
        y: Second sequence

    Returns:
        Length of LCS
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Use space-efficient LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def compute_rouge(hypothesis: str, reference: str) -> RougeMetrics:
    """Compute ROUGE metrics between hypothesis and reference.

    This implementation follows the standard ROUGE computation:
    - Tokenizes text into words
    - Computes unigram (ROUGE-1), bigram (ROUGE-2), and LCS (ROUGE-L) overlap
    - Returns precision, recall, and F1 for each

    Args:
        hypothesis: Generated summary
        reference: Ground truth summary

    Returns:
        RougeMetrics with all scores
    """
    # Try to use rouge-score library if available (more accurate)
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)

        return RougeMetrics(
            rouge_1_precision=scores["rouge1"].precision,
            rouge_1_recall=scores["rouge1"].recall,
            rouge_1_f1=scores["rouge1"].fmeasure,
            rouge_2_precision=scores["rouge2"].precision,
            rouge_2_recall=scores["rouge2"].recall,
            rouge_2_f1=scores["rouge2"].fmeasure,
            rouge_l_precision=scores["rougeL"].precision,
            rouge_l_recall=scores["rougeL"].recall,
            rouge_l_f1=scores["rougeL"].fmeasure,
        )

    except ImportError:
        # Fallback to simple implementation
        pass

    # Simple implementation
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return RougeMetrics()

    # ROUGE-1 (unigrams)
    hyp_unigrams = set(hyp_tokens)
    ref_unigrams = set(ref_tokens)
    overlap_1 = len(hyp_unigrams & ref_unigrams)
    r1_p, r1_r, r1_f = _compute_precision_recall_f1(len(hyp_unigrams), len(ref_unigrams), overlap_1)

    # ROUGE-2 (bigrams)
    hyp_bigrams = set(_get_ngrams(hyp_tokens, 2))
    ref_bigrams = set(_get_ngrams(ref_tokens, 2))
    overlap_2 = len(hyp_bigrams & ref_bigrams)
    r2_p, r2_r, r2_f = _compute_precision_recall_f1(len(hyp_bigrams), len(ref_bigrams), overlap_2)

    # ROUGE-L (LCS)
    lcs_len = _lcs_length(hyp_tokens, ref_tokens)
    rl_p, rl_r, rl_f = _compute_precision_recall_f1(len(hyp_tokens), len(ref_tokens), lcs_len)

    return RougeMetrics(
        rouge_1_precision=r1_p,
        rouge_1_recall=r1_r,
        rouge_1_f1=r1_f,
        rouge_2_precision=r2_p,
        rouge_2_recall=r2_r,
        rouge_2_f1=r2_f,
        rouge_l_precision=rl_p,
        rouge_l_recall=rl_r,
        rouge_l_f1=rl_f,
    )


def aggregate_rouge_metrics(metrics_list: list[RougeMetrics]) -> dict[str, float]:
    """Aggregate ROUGE metrics across multiple samples.

    Args:
        metrics_list: List of RougeMetrics objects

    Returns:
        Dictionary with mean values for each metric
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)

    return {
        "rouge_1_precision": sum(m.rouge_1_precision for m in metrics_list) / n,
        "rouge_1_recall": sum(m.rouge_1_recall for m in metrics_list) / n,
        "rouge_1_f1": sum(m.rouge_1_f1 for m in metrics_list) / n,
        "rouge_2_precision": sum(m.rouge_2_precision for m in metrics_list) / n,
        "rouge_2_recall": sum(m.rouge_2_recall for m in metrics_list) / n,
        "rouge_2_f1": sum(m.rouge_2_f1 for m in metrics_list) / n,
        "rouge_l_precision": sum(m.rouge_l_precision for m in metrics_list) / n,
        "rouge_l_recall": sum(m.rouge_l_recall for m in metrics_list) / n,
        "rouge_l_f1": sum(m.rouge_l_f1 for m in metrics_list) / n,
    }
