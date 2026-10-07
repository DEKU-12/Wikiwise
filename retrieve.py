"""
Retrieval: hybrid search (vector + keyword) with permission filtering and
cross-encoder reranking against the pgvector `chunks` table.

Pipeline per query:
  query -> [vector search top-N] + [keyword search top-N]   (permission-filtered)
        -> Reciprocal Rank Fusion -> fused candidates
        -> cross-encoder rerank -> final top-K

Requires the wikiwise-pgvector Docker container running on localhost:5433.
"""

import torch
from langsmith import traceable
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

import psycopg
from load_vectordb import DB_DSN

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# BGE's convention: queries get this instruction prefix, documents don't.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CANDIDATES_PER_ARM = 30
RRF_K = 60
FINAL_K = 5

ROLES = {
    "employee": {"can_see_restricted": False},
    "admin": {"can_see_restricted": True},
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _trace_pg_search_inputs(inputs: dict) -> dict:
    # conn isn't serializable; query_vector is a 384-float array, useless to
    # log — keep just what's meaningful for reading a trace.
    out = {"can_see_restricted": inputs.get("can_see_restricted"), "k": inputs.get("k")}
    if "query_text" in inputs:
        out["query_text"] = inputs["query_text"]
    return out


@traceable(name="pgvector_search", run_type="retriever", process_inputs=_trace_pg_search_inputs)
def vector_search(conn, query_vector, can_see_restricted: bool, k: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT chunk_id, title, department, source_url, source_path, text,
               1 - (embedding <=> %(qv)s) AS score
        FROM chunks
        WHERE restricted = false OR %(can_see_restricted)s
        ORDER BY embedding <=> %(qv)s
        LIMIT %(k)s
        """,
        {"qv": query_vector, "can_see_restricted": can_see_restricted, "k": k},
    ).fetchall()
    return rows


def _trace_kw_search_inputs(inputs: dict) -> dict:
    return {
        "query_text": inputs.get("query_text"),
        "can_see_restricted": inputs.get("can_see_restricted"),
        "k": inputs.get("k"),
    }


@traceable(name="postgres_fulltext_search", run_type="retriever", process_inputs=_trace_kw_search_inputs)
def keyword_search(conn, query_text: str, can_see_restricted: bool, k: int) -> list[dict]:
    # plainto_tsquery ANDs every significant word together ('a' & 'b' & 'c' & ...),
    # which almost never matches for a full natural-language question (a chunk
    # would need to contain every single word). We want OR semantics instead —
    # match chunks containing ANY of the terms, ranked by how many/how rare the
    # matches are via ts_rank — so we build the AND query first (to get
    # Postgres's stemming/stopword handling for free) then swap '&' for '|'.
    rows = conn.execute(
        """
        SELECT chunk_id, title, department, source_url, source_path, text,
               ts_rank(tsv, query) AS score
        FROM chunks,
             to_tsquery(
                 'english',
                 replace(plainto_tsquery('english', %(q)s)::text, '&', '|')
             ) query
        WHERE tsv @@ query AND (restricted = false OR %(can_see_restricted)s)
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        {"q": query_text, "can_see_restricted": can_see_restricted, "k": k},
    ).fetchall()
    return rows


@traceable(name="rrf_fuse", run_type="tool")
def rrf_fuse(result_lists: list[list[dict]], k_const: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = {}
    rows_by_id: dict[str, dict] = {}
    for results in result_lists:
        for rank, row in enumerate(results, start=1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
            rows_by_id.setdefault(cid, row)
    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [rows_by_id[cid] for cid in ranked_ids]


def _trace_rerank_inputs(inputs: dict) -> dict:
    return {
        "query": inputs.get("query"),
        "num_candidates": len(inputs.get("candidates") or []),
        "top_k": inputs.get("top_k"),
    }


@traceable(name="cross_encoder_rerank", run_type="tool", process_inputs=_trace_rerank_inputs)
def rerank(reranker: CrossEncoder, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
    results = []
    for candidate, score in ranked[:top_k]:
        results.append({**candidate, "rerank_score": float(score)})
    return results


def _trace_search_inputs(inputs: dict) -> dict:
    # conn/embed_model/reranker aren't serializable/useful in a trace —
    # only log what's actually interesting about the call.
    return {"query": inputs.get("query"), "can_see_restricted": inputs.get("can_see_restricted")}


@traceable(
    name="embed_query",
    run_type="embedding",
    process_inputs=lambda i: {"query": i.get("query")},
    process_outputs=lambda o: {"dimensions": len(o)},
)
def embed_query(embed_model: SentenceTransformer, query: str):
    return embed_model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]


@traceable(name="hybrid_search", run_type="retriever", process_inputs=_trace_search_inputs)
def search(
    conn,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    query: str,
    can_see_restricted: bool,
) -> list[dict]:
    query_vector = embed_query(embed_model, query)

    vec_results = vector_search(conn, query_vector, can_see_restricted, CANDIDATES_PER_ARM)
    kw_results = keyword_search(conn, query, can_see_restricted, CANDIDATES_PER_ARM)

    fused = rrf_fuse([vec_results, kw_results])
    return rerank(reranker, query, fused, FINAL_K)


def print_results(results: list[dict]):
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, start=1):
        snippet = r["text"][:200].replace("\n", " ")
        print(f"\n[{i}] score={r['rerank_score']:.3f}  {r['title']}  ({r['department']})")
        print(f"    {r['source_url']}")
        print(f"    {snippet}...")


def choose_role() -> dict:
    role_names = list(ROLES)
    print(f"Roles: {', '.join(role_names)}")
    while True:
        choice = input(f"Pick a role [{role_names[0]}]: ").strip() or role_names[0]
        if choice in ROLES:
            return ROLES[choice]
        print(f"Unknown role {choice!r}, choose from {role_names}")


def main():
    device = pick_device()
    print(f"Loading models on device={device} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

    role = choose_role()

    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        register_vector(conn)
        # Without this, a permission-filtered vector search (WHERE restricted
        # = false) can silently return fewer than the requested candidate
        # count when the true nearest neighbors cluster in a restricted
        # department — the HNSW scan stops at a fixed candidate window before
        # applying the filter. relaxed_order makes it keep searching until it
        # actually finds enough matches that pass the filter.
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        print("\nType a question (blank line to quit).")
        while True:
            query = input("\n> ").strip()
            if not query:
                break
            results = search(conn, embed_model, reranker, query, role["can_see_restricted"])
            print_results(results)


if __name__ == "__main__":
    main()
