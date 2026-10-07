"""
Embed every chunk in chunks.jsonl using a local sentence-transformers model
(BAAI/bge-small-en-v1.5) and load the vectors straight into the pgvector
`chunks` table.

Runs entirely on your machine — no API key, nothing leaves your laptop.
Requires the wikiwise-pgvector Docker container running on localhost:5433.
"""

import json
import time
from pathlib import Path

import torch
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

import psycopg
from load_vectordb import DB_DSN, create_schema, upsert_chunks

CHUNKS_FILE = Path("chunks.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    device = pick_device()
    print(f"Loading {MODEL_NAME} on device={device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE} ...")
    start = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/sec)")
    print(f"Embedding shape: {vectors.shape} ({vectors.shape[1]} dimensions)")

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    print("Loading into pgvector...")
    with psycopg.connect(DB_DSN) as conn:
        create_schema(conn)
        register_vector(conn)
        upsert_chunks(conn, chunks_by_id, ids, vectors)
        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"chunks table now has {count} rows")


if __name__ == "__main__":
    main()
