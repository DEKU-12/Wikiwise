"""
Streamlit front end for the Wikiwise RAG pipeline: pick a role, ask a
question, get a grounded answer with citations from the GitLab handbook.

Run with: streamlit run app.py
"""

import os

import anthropic
import psycopg
import streamlit as st
from dotenv import load_dotenv
from langsmith import trace
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

from wikiwise.generate import build_prompt, generate_answer
from wikiwise.load_vectordb import DB_DSN
from wikiwise.retrieve import (
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    ROLES,
    pick_device,
    search,
)

load_dotenv()

st.set_page_config(page_title="Wikiwise", page_icon="📘", layout="wide")


@st.cache_resource(show_spinner="Loading embedding + reranker models...")
def load_models():
    device = pick_device()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)
    return embed_model, reranker


def database_url() -> str:
    # Streamlit Cloud provides DATABASE_URL via st.secrets; locally there's no
    # secrets.toml, so fall back to .env / the local default in load_vectordb.
    try:
        return st.secrets["DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        return DB_DSN


def get_connection():
    conn = psycopg.connect(database_url(), row_factory=dict_row)
    register_vector(conn)
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    return conn


def retrieve_chunks(question: str, can_see_restricted: bool) -> list[dict]:
    embed_model, reranker = load_models()
    with get_connection() as conn:
        return search(conn, embed_model, reranker, question, can_see_restricted)


def render_sources(chunks: list[dict]):
    with st.expander(f"Sources ({len(chunks)})"):
        for i, c in enumerate(chunks, start=1):
            restricted_tag = " 🔒" if c.get("restricted") else ""
            st.markdown(
                f"**[{i}] {c['title']}** — *{c['department']}*{restricted_tag}  \n"
                f"[{c['source_url']}]({c['source_url']})"
            )


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    # Bring your own key: visitors use their own Anthropic key, held only in
    # their session. Falls back to ANTHROPIC_API_KEY for local runs.
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="Used only for this browser session, never stored. Get one at console.anthropic.com.",
    ) or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.info("Enter your Anthropic API key to start asking questions.")
    role_name = st.selectbox("Role", list(ROLES.keys()), index=0)
    can_see_restricted = ROLES[role_name]["can_see_restricted"]
    st.caption(
        "Can see restricted departments (finance, legal, people-group, etc.)"
        if can_see_restricted
        else "Cannot see restricted departments"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Main ---
st.title("📘 Wikiwise")
st.caption("Permission-aware RAG assistant over the GitLab team handbook")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about the GitLab handbook...", disabled=not api_key)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        client = anthropic.Anthropic(api_key=api_key)
        with trace(
            name="wikiwise_query",
            run_type="chain",
            inputs={"question": question, "can_see_restricted": can_see_restricted},
        ):
            with st.spinner("Retrieving..."):
                chunks = retrieve_chunks(question, can_see_restricted)

            if not chunks:
                answer = "No relevant context found in the handbook for this question."
                st.markdown(answer)
            else:
                prompt = build_prompt(question, chunks)
                try:
                    answer = st.write_stream(generate_answer(client, prompt))
                except anthropic.AuthenticationError:
                    answer = "That Anthropic API key was rejected. Check it in the sidebar and try again."
                    st.error(answer)

        if chunks:
            render_sources(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
