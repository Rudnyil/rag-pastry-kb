"""Chroma vector store wrapper (persistent, local)."""
from __future__ import annotations
from pathlib import Path
import chromadb
from embeddings import embedder_name, get_embedding_function

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"


def get_collection(path: Path = DB_PATH):
    client = chromadb.PersistentClient(path=str(path))
    # One collection per embedder, so vectors from different models are never mixed.
    # Cosine distance: smaller is more similar.
    return client.get_or_create_collection(
        name=f"pastry_kb_{embedder_name()}",
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
