"""
Retrieval evaluation against golden_dataset.jsonl.

Two independent checks:
1. Retrieval quality — run each golden question as a role that CAN see its
   answer (so permission filtering never blocks the correct chunk), then
   measure:
     - Recall@k / Precision@k / MRR (deterministic, exact chunk_id match)
     - Contextual Precision / Relevancy / Recall @5 (DeepEval, qwen2.5:7b judge)
2. Permission-leakage validation — re-run only the restricted-department
   questions as an `employee` (cannot see restricted content) and confirm
   zero restricted chunks ever appear in the results.

Requires: wikiwise-pgvector running, Ollama running with qwen2.5:7b pulled
and configured via `deepeval set-ollama` (already done).
"""

import json
from pathlib import Path

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

import psycopg
from wikiwise.load_vectordb import DB_DSN
from wikiwise.retrieve import (
    CANDIDATES_PER_ARM,
    EMBED_MODEL_NAME,
    QUERY_INSTRUCTION,
    RERANK_MODEL_NAME,
    keyword_search,
    pick_device,
    rerank,
    rrf_fuse,
    vector_search,
)

GOLDEN_FILE = Path("evals/golden_dataset.jsonl")
TOP_K = 10  # retrieve this many, so we can compute recall/precision at k=1,3,5,10
JUDGED_K = 5  # how many of those go to the DeepEval-judged contextual metrics
RECALL_KS = [1, 3, 5, 10]


def load_golden() -> list[dict]:
    with GOLDEN_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve_top_k(conn, embed_model, reranker, query: str, can_see_restricted: bool, k: int) -> list[dict]:
    query_vector = embed_model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]
    vec_results = vector_search(conn, query_vector, can_see_restricted, CANDIDATES_PER_ARM)
    kw_results = keyword_search(conn, query, can_see_restricted, CANDIDATES_PER_ARM)
    fused = rrf_fuse([vec_results, kw_results])
    return rerank(reranker, query, fused, k)


def rank_of_target(results: list[dict], target_chunk_id: str) -> int | None:
    for i, r in enumerate(results, start=1):
        if r["chunk_id"] == target_chunk_id:
            return i
    return None


JUDGE_RETRIES = 2


def _measure_with_retry(metric, tc, label: str, question_id: str) -> float | None:
    """DeepEval's judge occasionally returns malformed JSON and raises instead
    of retrying cleanly. Retry a couple of times, then skip just this metric
    for this question rather than losing the whole run."""
    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            metric.measure(tc)
            return metric.score
        except Exception as e:
            print(f"    [warn] {label} failed on {question_id} (attempt {attempt}): {e}")
    print(f"    [skip] {label} skipped for {question_id} after {JUDGE_RETRIES} failed attempts")
    return None


def evaluate_retrieval_quality(conn, embed_model, reranker, golden: list[dict]) -> dict:
    recall_hits = {k: 0 for k in RECALL_KS}
    precision_sum = {k: 0.0 for k in RECALL_KS}
    reciprocal_ranks = []

    ctx_precision = ContextualPrecisionMetric()
    ctx_relevancy = ContextualRelevancyMetric()
    ctx_recall = ContextualRecallMetric()
    precision_scores, relevancy_scores, recall_scores = [], [], []

    for i, item in enumerate(golden, start=1):
        print(f"  [{i}/{len(golden)}] {item['question_id']}: {item['question'][:60]}")
        results = retrieve_top_k(
            conn, embed_model, reranker, item["question"], can_see_restricted=True, k=TOP_K
        )

        rank = rank_of_target(results, item["chunk_id"])
        for k in RECALL_KS:
            hit = rank is not None and rank <= k
            recall_hits[k] += int(hit)
            precision_sum[k] += (1.0 / k) if hit else 0.0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        judged_texts = [r["text"] for r in results[:JUDGED_K]]
        tc = LLMTestCase(
            input=item["question"],
            expected_output=item["reference_answer"],
            retrieval_context=judged_texts,
        )
        p = _measure_with_retry(ctx_precision, tc, "ContextualPrecision", item["question_id"])
        r = _measure_with_retry(ctx_relevancy, tc, "ContextualRelevancy", item["question_id"])
        c = _measure_with_retry(ctx_recall, tc, "ContextualRecall", item["question_id"])
        if p is not None:
            precision_scores.append(p)
        if r is not None:
            relevancy_scores.append(r)
        if c is not None:
            recall_scores.append(c)

    n = len(golden)
    return {
        "recall_at_k": {k: recall_hits[k] / n for k in RECALL_KS},
        "precision_at_k": {k: precision_sum[k] / n for k in RECALL_KS},
        "mrr": sum(reciprocal_ranks) / n,
        "contextual_precision": sum(precision_scores) / len(precision_scores) if precision_scores else None,
        "contextual_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None,
        "contextual_recall": sum(recall_scores) / len(recall_scores) if recall_scores else None,
        "contextual_precision_n": len(precision_scores),
        "contextual_relevancy_n": len(relevancy_scores),
        "contextual_recall_n": len(recall_scores),
    }


def evaluate_permission_leakage(conn, embed_model, reranker, golden: list[dict]) -> dict:
    restricted_items = [g for g in golden if g["restricted"]]
    leaks = []

    for i, item in enumerate(restricted_items, start=1):
        print(f"  [{i}/{len(restricted_items)}] {item['question_id']}: {item['question'][:60]}")
        results = retrieve_top_k(
            conn, embed_model, reranker, item["question"], can_see_restricted=False, k=TOP_K
        )
        leaked_chunks = [r["chunk_id"] for r in results if r.get("restricted")]
        if leaked_chunks:
            leaks.append({"question_id": item["question_id"], "leaked_chunk_ids": leaked_chunks})

    return {
        "restricted_questions_tested": len(restricted_items),
        "questions_with_leaks": len(leaks),
        "leak_rate": len(leaks) / len(restricted_items) if restricted_items else 0.0,
        "details": leaks,
    }


def main():
    golden = load_golden()
    print(f"Loaded {len(golden)} golden questions")

    device = pick_device()
    print(f"Loading retrieval models on device={device} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        register_vector(conn)
        conn.execute("SET hnsw.iterative_scan = relaxed_order")

        print("\n=== Retrieval quality (admin role, all questions) ===")
        quality = evaluate_retrieval_quality(conn, embed_model, reranker, golden)

        print("\n=== Permission-leakage validation (employee role, restricted questions) ===")
        leakage = evaluate_permission_leakage(conn, embed_model, reranker, golden)

    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY")
    for k in RECALL_KS:
        print(f"  Recall@{k}:    {quality['recall_at_k'][k]:.1%}")
    for k in RECALL_KS:
        print(f"  Precision@{k}: {quality['precision_at_k'][k]:.1%}")
    print(f"  MRR:               {quality['mrr']:.3f}")

    def fmt(score, n):
        return f"{score:.3f} (n={n}/{len(golden)})" if score is not None else f"N/A (0/{len(golden)} succeeded)"

    print(f"  Contextual Precision @{JUDGED_K}: {fmt(quality['contextual_precision'], quality['contextual_precision_n'])}")
    print(f"  Contextual Relevancy @{JUDGED_K}: {fmt(quality['contextual_relevancy'], quality['contextual_relevancy_n'])}")
    print(f"  Contextual Recall @{JUDGED_K}:    {fmt(quality['contextual_recall'], quality['contextual_recall_n'])}")

    print("\nPERMISSION-LEAKAGE VALIDATION")
    print(f"  Restricted questions tested: {leakage['restricted_questions_tested']}")
    print(f"  Questions with a leak:       {leakage['questions_with_leaks']}")
    print(f"  Leak rate:                   {leakage['leak_rate']:.1%}")
    if leakage["details"]:
        print("  Details:")
        for d in leakage["details"]:
            print(f"    {d['question_id']}: leaked {d['leaked_chunk_ids']}")

    with open("evals/results/eval_retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump({"quality": quality, "leakage": leakage}, f, indent=2)
    print("\nSaved full results to evals/results/eval_retrieval_results.json")


if __name__ == "__main__":
    main()
