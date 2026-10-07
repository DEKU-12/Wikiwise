"""
Build a small entity-relationship knowledge graph from a sample of the
GitLab handbook, using Claude to extract entities (people, teams,
departments, tools, policies) and relationships between them from each
page's actual text.

Sample: all pages from leadership/eba/ceo (small, known to have named
people/roles) plus a random sample from company (working-group tables with
real names/roles). Output: knowledge_graph.json (nodes + edges).
"""

import json
import random
from pathlib import Path

import anthropic
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from wikiwise.load_vectordb import DB_DSN

load_dotenv()

OUTPUT_FILE = Path("data/knowledge_graph.json")
MODEL_NAME = "claude-sonnet-5"

FULL_DEPARTMENTS = ["leadership", "eba", "ceo"]
SAMPLE_DEPARTMENT = "company"
SAMPLE_SIZE = 30
RANDOM_SEED = 42

ENTITY_TYPES = ["Person", "Team", "Department", "Tool", "Policy", "Process"]

EXTRACTION_PROMPT = """You are extracting a knowledge graph from one page of an internal company handbook.

Identify:
1. Entities explicitly named in the text: real people (by name), teams/working groups, departments, tools/systems, policies, or processes.
2. Relationships between those entities that are explicitly and clearly stated (e.g. "is DRI of", "is a member of", "manages", "reports to", "uses", "owns", "is responsible for", "is Executive Sponsor of").

Rules:
- Only extract what is explicitly stated. Do not infer or guess relationships that aren't directly written.
- Use each entity's name exactly as written (don't paraphrase).
- If the page has no clear entities or relationships, return empty lists.

Page title: {title}
Page text:
\"\"\"
{text}
\"\"\"

Respond with ONLY a JSON object, no other text:
{{"entities": [{{"name": "...", "type": "Person|Team|Department|Tool|Policy|Process"}}], "relationships": [{{"source": "...", "relation": "...", "target": "..."}}]}}"""


def get_pages(conn) -> list[dict]:
    """One row per page (chunks joined back together), for the sampled departments."""
    rows = conn.execute(
        """
        SELECT source_path, title, department, source_url,
               string_agg(text, E'\\n\\n' ORDER BY chunk_index) AS text
        FROM chunks
        WHERE department = ANY(%(full_depts)s)
        GROUP BY source_path, title, department, source_url
        """,
        {"full_depts": FULL_DEPARTMENTS},
    ).fetchall()

    sample_pages = conn.execute(
        """
        SELECT source_path, title, department, source_url,
               string_agg(text, E'\\n\\n' ORDER BY chunk_index) AS text
        FROM chunks
        WHERE department = %(dept)s
        GROUP BY source_path, title, department, source_url
        ORDER BY source_path
        """,
        {"dept": SAMPLE_DEPARTMENT},
    ).fetchall()
    random.Random(RANDOM_SEED).shuffle(sample_pages)
    sample_pages = sample_pages[:SAMPLE_SIZE]

    return list(rows) + sample_pages


def extract_graph(client: anthropic.Anthropic, page: dict) -> dict | None:
    prompt = EXTRACTION_PROMPT.format(title=page["title"], text=page["text"][:6000])
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        return None  # e.g. hit max_tokens while still thinking, no text produced
    raw = text_blocks[0].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.removesuffix("```").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def normalize_key(name: str, type_: str) -> str:
    return f"{type_}:{name.strip().lower()}"


def main():
    client = anthropic.Anthropic()

    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        pages = get_pages(conn)
    print(f"Processing {len(pages)} pages ({', '.join(FULL_DEPARTMENTS)} full, "
          f"{SAMPLE_SIZE} sampled from {SAMPLE_DEPARTMENT})")

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    failed = 0

    for i, page in enumerate(pages, start=1):
        print(f"  [{i}/{len(pages)}] {page['department']}: {page['title'][:60]}")
        graph = extract_graph(client, page)
        if graph is None:
            failed += 1
            continue

        for entity in graph.get("entities", []):
            name, type_ = entity.get("name"), entity.get("type")
            if not name or type_ not in ENTITY_TYPES:
                continue
            key = normalize_key(name, type_)
            if key not in nodes:
                nodes[key] = {"id": key, "name": name, "type": type_, "sources": []}
            nodes[key]["sources"].append({"title": page["title"], "url": page["source_url"]})

        for rel in graph.get("relationships", []):
            source, relation, target = rel.get("source"), rel.get("relation"), rel.get("target")
            if not source or not relation or not target:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "page_title": page["title"],
                    "page_url": page["source_url"],
                }
            )

    print(f"\nExtracted {len(nodes)} entities and {len(edges)} relationships ({failed} pages failed to parse)")

    graph_data = {"nodes": list(nodes.values()), "edges": edges}
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
