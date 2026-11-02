import json
from pathlib import Path
import chunking, store, ingest, ask

DATA = Path(__file__).resolve().parent.parent / "data"


def test_chunks_follow_sections():
    chunks = chunking.split_markdown((DATA / "pastry_shop_policies.md").read_text(encoding="utf-8"), "x.md")
    sections = {c["metadata"]["section"] for c in chunks}
    assert "Delivery" in sections and "Prices" in sections
    assert all(len(c["text"]) <= 1100 for c in chunks)


def test_ingest_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "db"))
    col = store.get_collection(str(tmp_path / "db"))
    chunks = ingest.load_chunks()
    for _ in range(2):
        col.upsert(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                   metadatas=[c["metadata"] for c in chunks])
    assert col.count() == len(chunks)


def test_retrieval_finds_delivery(tmp_path):
    col = store.get_collection(str(tmp_path / "db"))
    chunks = ingest.load_chunks()
    col.upsert(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
               metadatas=[c["metadata"] for c in chunks])
    res = col.query(query_texts=["Do you deliver on Sunday?"], n_results=3)
    assert "Delivery" in [m["section"] for m in res["metadatas"][0]]
