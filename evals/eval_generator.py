"""
Generator evaluation against golden_dataset.jsonl.

For each golden question: retrieve with the real production pipeline
(retrieve.py, admin role so permission filtering never blocks the answer),
generate an answer with the real production generator (generate.py, Claude
Sonnet 5), then score the answer with DeepEval's judge (Claude Haiku 4.5):

  - Faithfulness    — does the answer stick to the retrieved context?
  - Answer Relevancy — does the answer actually address the question?

Requires: wikiwise-pgvector running, ANTHROPIC_API_KEY in .env.
"""

import json
from pathlib import Path

import anthropic
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

import psycopg
from evals.eval_retrieval import _measure_with_retry
from wikiwise.generate import build_prompt, generate_answer
from wikiwise.load_vectordb import DB_DSN
from wikiwise.retrieve import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device, search

load_dotenv()

GOLDEN_FILE = Path("evals/golden_dataset.jsonl")
RESULTS_FILE = Path("evals/results/eval_generator_results.json")


def load_golden() -> list[dict]:
    with GOLDEN_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_generation(conn, client, embed_model, reranker, golden: list[dict]) -> dict:
    faithfulness_metric = FaithfulnessMetric()
    relevancy_metric = AnswerRelevancyMetric()
    faithfulness_scores, relevancy_scores = [], []
    per_question = []

    for i, item in enumerate(golden, start=1):
        print(f"  [{i}/{len(golden)}] {item['question_id']}: {item['question'][:60]}")

        chunks = search(conn, embed_model, reranker, item["question"], can_see_restricted=True)
        prompt = build_prompt(item["question"], chunks)
        answer = generate_answer(client, prompt)

        tc = LLMTestCase(
            input=item["question"],
            actual_output=answer,
            retrieval_context=[c["text"] for c in chunks],
        )
        f_score = _measure_with_retry(faithfulness_metric, tc, "Faithfulness", item["question_id"])
        r_score = _measure_with_retry(relevancy_metric, tc, "AnswerRelevancy", item["question_id"])
        if f_score is not None:
            faithfulness_scores.append(f_score)
        if r_score is not None:
            relevancy_scores.append(r_score)

        per_question.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": answer,
                "faithfulness": f_score,
                "answer_relevancy": r_score,
            }
        )

    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "faithfulness_n": len(faithfulness_scores),
        "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None,
        "answer_relevancy_n": len(relevancy_scores),
        "per_question": per_question,
    }


def main():
    golden = load_golden()
    print(f"Loaded {len(golden)} golden questions")

    client = anthropic.Anthropic()
    device = pick_device()
    print(f"Loading retrieval models on device={device} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        register_vector(conn)
        conn.execute("SET hnsw.iterative_scan = relaxed_order")

        print("\n=== Generator evaluation (retrieve -> generate -> judge) ===")
        results = evaluate_generation(conn, client, embed_model, reranker, golden)

    n = len(golden)

    def fmt(score, k):
        return f"{score:.3f} (n={k}/{n})" if score is not None else f"N/A (0/{n} succeeded)"

    print("\n" + "=" * 60)
    print("GENERATOR QUALITY")
    print(f"  Faithfulness:     {fmt(results['faithfulness'], results['faithfulness_n'])}")
    print(f"  Answer Relevancy: {fmt(results['answer_relevancy'], results['answer_relevancy_n'])}")

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results (incl. per-question answers) to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
