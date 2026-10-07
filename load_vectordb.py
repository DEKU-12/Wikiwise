"""
Load chunks.jsonl + embeddings.npy into Postgres/pgvector.

Creates the `chunks` table (with an HNSW index for cosine similarity search)
and upserts every chunk with its embedding and metadata, so retrieval can
later query it directly with SQL (including permission filters on
department/restricted).

Connects to DATABASE_URL (e.g. a Neon connection string) if set, otherwise
the local wikiwise-pgvector Docker container on localhost:5433.
"""

import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

CHUNKS_FILE = Path("chunks.jsonl")
EMBEDDINGS_FILE = Path("embeddings.npy")
IDS_FILE = Path("embedding_ids.json")

DB_DSN = os.getenv("DATABASE_URL", "postgresql://wikiwise:wikiwise@localhost:5433/wikiwise")
EMBEDDING_DIM = 384
BATCH_SIZE = 500


def load_chunks() -> dict[str, dict]:
    chunks = {}
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


def load_embeddings() -> tuple[list[str], np.ndarray]:
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_FILE)
    return ids, vectors


def create_schema(conn: psycopg.Connection):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id     TEXT PRIMARY KEY,
            text         TEXT NOT NULL,
            heading      TEXT,
            chunk_index  INTEGER,
            title        TEXT,
            department   TEXT,
            source_url   TEXT,
            source_path  TEXT,
            restricted   BOOLEAN,
            embedding    VECTOR({EMBEDDING_DIM})
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_department_idx ON chunks (department)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_restricted_idx ON chunks (restricted)"
    )
    # full-text search column for the keyword arm of hybrid search
    conn.execute(
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)")
    conn.commit()


def upsert_chunks(conn: psycopg.Connection, chunks: dict[str, dict], ids: list[str], vectors: np.ndarray):
    rows = []
    for chunk_id, vector in zip(ids, vectors):
        c = chunks[chunk_id]
        rows.append(
            (
                c["chunk_id"],
                c["text"],
                c["heading"],
                c["chunk_index"],
                c["title"],
                c["department"],
                c["source_url"],
                c["source_path"],
                c["restricted"],
                vector,
            )
        )

    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            cur.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, text, heading, chunk_index, title, department,
                     source_url, source_path, restricted, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    heading = EXCLUDED.heading,
                    chunk_index = EXCLUDED.chunk_index,
                    title = EXCLUDED.title,
                    department = EXCLUDED.department,
                    source_url = EXCLUDED.source_url,
                    source_path = EXCLUDED.source_path,
                    restricted = EXCLUDED.restricted,
                    embedding = EXCLUDED.embedding
                """,
                batch,
            )
            print(f"  upserted {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")
    conn.commit()


def main():
    chunks = load_chunks()
    ids, vectors = load_embeddings()
    print(f"Loaded {len(chunks)} chunks and {len(ids)} embeddings ({vectors.shape[1]} dims)")

    with psycopg.connect(DB_DSN) as conn:
        print("Creating schema (table + indexes) if not present...")
        create_schema(conn)
        register_vector(conn)

        print("Upserting rows...")
        upsert_chunks(conn, chunks, ids, vectors)

        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        restricted_count = conn.execute(
            "SELECT count(*) FROM chunks WHERE restricted"
        ).fetchone()[0]
        print(f"\nchunks table now has {count} rows ({restricted_count} restricted)")


if __name__ == "__main__":
    main()
