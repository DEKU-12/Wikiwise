"""
Ingest the GitLab Handbook into clean, chunked, metadata-tagged text.

Step 1: discover every markdown page under handbook/content/handbook/.
Step 2: extract front matter + body, and clean the body into plain,
        embedding-friendly text (strip Hugo shortcodes/diagrams/HTML,
        keep the actual words they carried).
Step 3: split each page into chunks with LangChain's Markdown-aware
        RecursiveCharacterTextSplitter, with metadata attached.

Output: chunks.jsonl, one JSON object per chunk, ready for embedding.
"""

import hashlib
import json
import re
from pathlib import Path

import frontmatter
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

HANDBOOK_DIR = Path("handbook/content/handbook")
OUTPUT_FILE = Path("data/chunks.jsonl")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

RESTRICTED_DEPARTMENTS = {
    "people-group",
    "people-policies",
    "total-rewards",
    "finance",
    "legal",
    "labor-and-employment-notices",
}

# ---------------------------------------------------------------------------
# Step 2: extract + clean
# ---------------------------------------------------------------------------

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MERMAID_RE = re.compile(r"```mermaid.*?```", re.DOTALL)
SHORTCODE_RE = re.compile(r"\{\{[%<].*?[%>]\}\}", re.DOTALL)
SHORTCODE_NAME_RE = re.compile(r"\{\{[%<]\s*([\w/-]+)")
QUOTED_ARG_RE = re.compile(r'"([^"]*)"')
LABEL_NAME_RE = re.compile(r'name="([^"]*)"')
HTML_TAG_RE = re.compile(r"<[^>]+>")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

# Shortcodes whose argument carries real, recoverable text.
TEXT_VALUE_SHORTCODES = {"member-by-name", "member-by-gitlab", "manager-by-report-name"}


def _replace_shortcode(match: re.Match) -> str:
    """Turn a single {{< ... >}} / {{% ... %}} tag into whatever plain text
    it was standing in for (a name, a label, yes/no), or drop it if the tag
    only renders dynamic content that isn't present in the raw markdown."""
    tag = match.group(0)
    name_match = SHORTCODE_NAME_RE.match(tag)
    name = name_match.group(1) if name_match else ""

    if name == "yes":
        return "Yes"
    if name == "no":
        return "No"
    if name in TEXT_VALUE_SHORTCODES:
        arg = QUOTED_ARG_RE.search(tag)
        return arg.group(1) if arg else ""
    if name == "label":
        label = LABEL_NAME_RE.search(tag)
        return label.group(1) if label else ""
    # team-by-*, group-by-*, kpi, youtube, engineering/*, note (open/close), etc.
    # render dynamic data or markup with nothing recoverable from source text.
    return ""


def clean_body(text: str) -> str:
    text = HTML_COMMENT_RE.sub("", text)
    text = MERMAID_RE.sub("[diagram omitted]", text)
    text = SHORTCODE_RE.sub(_replace_shortcode, text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = HTML_TAG_RE.sub("", text)
    text = IMAGE_RE.sub(r"\1", text)
    text = LINK_RE.sub(r"\1", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_and_clean(path: Path) -> dict | None:
    post = frontmatter.load(path)
    text = clean_body(post.content)
    if not text:
        return None

    title = post.get("title") or path.stem.replace("-", " ").title()
    department = path.relative_to(HANDBOOK_DIR).parts[0]

    return {
        "source_path": str(path),
        "source_url": source_url_for(path),
        "title": title,
        "department": department,
        "restricted": department in RESTRICTED_DEPARTMENTS,
        "text": text,
    }


def source_url_for(path: Path) -> str:
    rel = path.relative_to(HANDBOOK_DIR)
    url_path = rel.parent if rel.name == "_index.md" else rel.with_suffix("")
    posix = url_path.as_posix()
    if posix == ".":
        return "https://handbook.gitlab.com/handbook/"
    return f"https://handbook.gitlab.com/handbook/{posix}/"


# ---------------------------------------------------------------------------
# Step 3: chunk (LangChain's RecursiveCharacterTextSplitter, Markdown-aware)
# ---------------------------------------------------------------------------

# from_language(Language.MARKDOWN, ...) gives the splitter a Markdown-specific
# separator priority list: it tries to cut on heading lines (#..######) first,
# then code fences / horizontal rules, then blank lines, then single newlines,
# then words, then characters as a last resort.
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

FIRST_HEADING_RE = re.compile(r"^#{2,6}\s+(.*)$", re.MULTILINE)
TINY_PIECE_SIZE = 50


def merge_tiny_pieces(pieces: list[str]) -> list[str]:
    """The splitter sometimes flushes a chunk right at a heading boundary,
    leaving a bare '### Heading' as its own tiny chunk. Fold anything under
    TINY_PIECE_SIZE into the next piece so headings stay with their body."""
    merged: list[str] = []
    pending = ""
    for piece in pieces:
        combined = f"{pending}\n\n{piece}" if pending else piece
        if len(combined) < TINY_PIECE_SIZE:
            pending = combined
            continue
        merged.append(combined)
        pending = ""
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending}"
        else:
            merged.append(pending)
    return merged


def chunk_page(record: dict) -> list[dict]:
    pieces = merge_tiny_pieces(splitter.split_text(record["text"]))

    chunks = []
    for chunk_index, piece in enumerate(pieces):
        # best-effort label: the first heading line the chunk starts under,
        # if any (purely informational, not used for splitting)
        heading_match = FIRST_HEADING_RE.search(piece)
        heading = heading_match.group(1).strip() if heading_match else ""

        chunk_id = hashlib.sha1(
            f"{record['source_path']}:{chunk_index}".encode()
        ).hexdigest()[:16]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": piece,
                "heading": heading,
                "chunk_index": chunk_index,
                "title": record["title"],
                "department": record["department"],
                "source_url": record["source_url"],
                "source_path": record["source_path"],
                "restricted": record["restricted"],
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    files = sorted(HANDBOOK_DIR.rglob("*.md"))
    print(f"Found {len(files)} markdown files under {HANDBOOK_DIR}")

    records = []
    skipped = 0
    for path in files:
        record = extract_and_clean(path)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"Extracted {len(records)} pages ({skipped} skipped as empty)")

    all_chunks = []
    for record in records:
        all_chunks.extend(chunk_page(record))

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    dept_counts: dict[str, int] = {}
    lengths = []
    restricted_count = 0
    for c in all_chunks:
        dept_counts[c["department"]] = dept_counts.get(c["department"], 0) + 1
        lengths.append(len(c["text"]))
        if c["restricted"]:
            restricted_count += 1

    print(f"\nWrote {len(all_chunks)} chunks to {OUTPUT_FILE}")
    print(f"Restricted chunks: {restricted_count} ({restricted_count / len(all_chunks):.1%})")
    print(
        f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars "
        f"(min {min(lengths)}, max {max(lengths)})"
    )
    print(f"Chunks under 40 chars (noise check): {sum(1 for l in lengths if l < 40)}")

    print("\nTop departments by chunk count:")
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:5d}  {dept}")


if __name__ == "__main__":
    main()
