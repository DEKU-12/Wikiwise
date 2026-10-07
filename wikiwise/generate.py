"""
Generation: retrieve chunks with retrieve.py's hybrid-search pipeline, then
ask Claude to answer using only that context, with numbered citations.

Requires ANTHROPIC_API_KEY set in .env (project root).
"""

import anthropic
import psycopg
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

from wikiwise.load_vectordb import DB_DSN
from wikiwise.retrieve import (
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    choose_role,
    pick_device,
    search,
)

load_dotenv()

GENERATOR_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are an internal assistant answering questions using ONLY the numbered context passages below, drawn from the GitLab Handbook.
Rules:
- Answer only using the provided context. Do not use outside knowledge.
- Cite sources inline using [1], [2], etc. matching the passage numbers.
- If the context does not contain enough information to answer, say so plainly instead of guessing.
- Be concise: answer directly in as few sentences as the question allows. Do not add background, caveats, or elaboration beyond what was asked."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] {c['title']} ({c['department']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def _trace_generate_inputs(inputs: dict) -> dict:
    return {"prompt": inputs.get("prompt")}  # client isn't serializable/useful in a trace


@traceable(
    name="claude_generate",
    run_type="llm",
    metadata={"ls_provider": "anthropic", "ls_model_name": GENERATOR_MODEL},
    process_inputs=_trace_generate_inputs,
    reduce_fn=lambda chunks: "".join(chunks),
)
def generate_answer(client: anthropic.Anthropic, prompt: str):
    """Streams the answer — yields text chunks as Claude generates them, so
    the caller can render them progressively instead of waiting for the
    full response. Doesn't reduce total latency, but time-to-first-token
    is much faster than time-to-full-response."""
    with client.messages.stream(
        model=GENERATOR_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream

        final_message = stream.get_final_message()
        # wrap_anthropic() would normally do this automatically, but it's
        # incompatible with the modern anthropic SDK (see retrieve.py/README
        # notes) — so we attach token usage manually. LangSmith uses this
        # (input_tokens/output_tokens + the ls_model_name set above) to
        # compute cost, as long as the model is in its pricing table.
        run = get_current_run_tree()
        if run is not None:
            run.set(
                usage_metadata={
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                    "total_tokens": final_message.usage.input_tokens
                    + final_message.usage.output_tokens,
                }
            )


def print_citations(chunks: list[dict]):
    print("\nSources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['title']} — {c['source_url']}")


def main():
    client = anthropic.Anthropic()

    device = pick_device()
    print(f"Loading retrieval models on device={device} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

    role = choose_role()

    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        register_vector(conn)
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        print("\nAsk a question (blank line to quit).")
        while True:
            query = input("\n> ").strip()
            if not query:
                break

            chunks = search(conn, embed_model, reranker, query, role["can_see_restricted"])
            if not chunks:
                print("No relevant context found.")
                continue

            prompt = build_prompt(query, chunks)
            print()
            for text in generate_answer(client, prompt):
                print(text, end="", flush=True)
            print()
            print_citations(chunks)


if __name__ == "__main__":
    main()
