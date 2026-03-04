"""
RAG Pipeline — KB Loader + Retriever

Production pattern for grounding agents in a domain-specific knowledge base.
Supports markdown files (chunked) with hybrid search fallback.

Usage:
    from rag.retriever import build_retriever
    retriever = build_retriever()
    docs = retriever.invoke("how do I redeem a PSN card?")
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_anthropic import AnthropicEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain_core.documents import Document

KB_DIR = Path(__file__).parent / "kb"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# ─── KB loader ────────────────────────────────────────────────────────────────

def load_kb(kb_dir: Path = KB_DIR) -> list[Document]:
    """Load all .md files from the KB directory into LangChain documents."""
    if not kb_dir.exists():
        return []

    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []

    for md_file in kb_dir.glob("**/*.md"):
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": str(md_file.relative_to(kb_dir))}]
        )
        docs.extend(chunks)

    return docs

# ─── Retriever ────────────────────────────────────────────────────────────────

_retriever_cache = None

def build_retriever(kb_dir: Path = KB_DIR, k: int = 3):
    """Build a FAISS retriever from the KB directory.

    Uses Anthropic embeddings. Caches the index after first build.
    In production: persist the FAISS index to disk or swap for Supabase pgvector.
    """
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    docs = load_kb(kb_dir)

    if not docs:
        # Return a dummy retriever if KB is empty (graceful degradation)
        class EmptyRetriever:
            def invoke(self, _): return []
        return EmptyRetriever()

    embeddings = AnthropicEmbeddings(
        model="voyage-3",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    _retriever_cache = vectorstore.as_retriever(search_kwargs={"k": k})
    return _retriever_cache
