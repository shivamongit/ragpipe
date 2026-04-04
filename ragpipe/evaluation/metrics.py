"""RAG evaluation metrics for retrieval quality and generation faithfulness."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragpipe.core import RetrievalResult


def hit_rate(results: list[RetrievalResult], relevant_doc_ids: set[str]) -> float:
    """Fraction of queries where at least one relevant document is retrieved.

    Returns 1.0 if any retrieved chunk belongs to a relevant document, else 0.0.
    """
    for r in results:
        if r.chunk.doc_id in relevant_doc_ids:
            return 1.0
    return 0.0


def mrr(results: list[RetrievalResult], relevant_doc_ids: set[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result.

    Returns 0.0 if no relevant document is found.
    """
    for i, r in enumerate(results):
        if r.chunk.doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    results: list[RetrievalResult], relevant_doc_ids: set[str], k: int = 5
) -> float:
    """Precision@K — fraction of top-k results that are relevant."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if r.chunk.doc_id in relevant_doc_ids)
    return relevant / len(top_k)


def recall_at_k(
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
    k: int = 5,
) -> float:
    """Recall@K — fraction of relevant documents found in top-k results."""
    if not relevant_doc_ids:
        return 0.0
    top_k = results[:k]
    found = set()
    for r in top_k:
        if r.chunk.doc_id in relevant_doc_ids:
            found.add(r.chunk.doc_id)
    return len(found) / len(relevant_doc_ids)


def ndcg_at_k(
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Measures ranking quality — relevant documents ranked higher contribute
    more to the score. Normalizes against the ideal ranking.
    """
    top_k = results[:k]
    if not top_k or not relevant_doc_ids:
        return 0.0

    dcg = 0.0
    for i, r in enumerate(top_k):
        rel = 1.0 if r.chunk.doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG: all relevant docs at the top
    n_relevant = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def map_at_k(
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
    k: int = 5,
) -> float:
    """Mean Average Precision at K.

    Computes the average of precision values at each relevant position
    in the top-k results. Rewards systems that rank relevant docs earlier.
    """
    top_k = results[:k]
    if not top_k or not relevant_doc_ids:
        return 0.0

    score = 0.0
    hits = 0
    for i, r in enumerate(top_k):
        if r.chunk.doc_id in relevant_doc_ids:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant_doc_ids), k) if relevant_doc_ids else 0.0


def rouge_l(answer: str, reference: str) -> dict[str, float]:
    """ROUGE-L score based on Longest Common Subsequence.

    Measures overlap between generated answer and reference answer.
    Returns precision, recall, and F1 scores.
    """
    answer_tokens = answer.lower().split()
    reference_tokens = reference.lower().split()

    if not answer_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m = len(answer_tokens)
    n = len(reference_tokens)

    # LCS using dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if answer_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def context_precision(
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
) -> float:
    """Context Precision — weighted precision that penalizes irrelevant chunks
    appearing before relevant ones.

    Used in RAGAS-style evaluation. Higher score means relevant chunks
    are ranked earlier in the retrieved results.
    """
    if not results or not relevant_doc_ids:
        return 0.0

    score = 0.0
    hits = 0
    for i, r in enumerate(results):
        if r.chunk.doc_id in relevant_doc_ids:
            hits += 1
            score += hits / (i + 1)

    return score / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def faithfulness_score(answer: str, source_texts: list[str]) -> dict[str, float]:
    """Simple faithfulness estimation based on n-gram overlap.

    Measures how much of the generated answer's content is grounded
    in the source texts. Returns unigram and bigram overlap ratios.

    For production use, consider LLM-based faithfulness evaluation
    (e.g., G-Eval, RAGAS) instead of this heuristic.
    """
    answer_tokens = answer.lower().split()
    if not answer_tokens:
        return {"unigram_overlap": 0.0, "bigram_overlap": 0.0}

    source_text = " ".join(source_texts).lower()
    source_tokens = source_text.split()
    source_token_set = set(source_tokens)

    # Unigram overlap
    matched_unigrams = sum(1 for t in answer_tokens if t in source_token_set)
    unigram_overlap = matched_unigrams / len(answer_tokens)

    # Bigram overlap
    def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    answer_bigrams = bigrams(answer_tokens)
    source_bigrams = bigrams(source_tokens)

    if not answer_bigrams:
        bigram_overlap = 0.0
    else:
        matched_bigrams = len(answer_bigrams & source_bigrams)
        bigram_overlap = matched_bigrams / len(answer_bigrams)

    return {
        "unigram_overlap": round(unigram_overlap, 4),
        "bigram_overlap": round(bigram_overlap, 4),
    }
