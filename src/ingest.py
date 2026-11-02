"""Read documents from data/, chunk them and store embeddings in Chroma."""
from __future__ import annotations
from pathlib import Path
from chunking import split_markdown
from store import get_collection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        chunks += split_markdown(path.read_text(encoding="utf-8"), source=path.name)
    return chunks


def main() -> None:
    col = get_collection()
    chunks = load_chunks()
    col.upsert(  # upsert: running ingest twice does not create duplicates
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"Stored {len(chunks)} chunks. Collection size: {col.count()}")


if __name__ == "__main__":
    main()
