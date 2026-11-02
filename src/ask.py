"""Ask a question: retrieve the best chunks, then (optionally) let an LLM answer from them."""
from __future__ import annotations
import argparse
import os
import requests
from dotenv import load_dotenv
from embeddings import MAX_DISTANCE, embedder_name
from store import get_collection

load_dotenv()
TOP_K = 3

PROMPT = """You answer questions about a pastry shop using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know based on the documents.
Keep the answer short. Mention the section name you used.

Context:
{context}

Question: {question}"""


def retrieve(question: str, k: int = TOP_K):
    res = get_collection().query(query_texts=[question], n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "section": meta["section"], "distance": dist})
    return hits


def answer(question: str) -> dict:
    hits = [h for h in retrieve(question) if h["distance"] <= MAX_DISTANCE[embedder_name()]]
    if not hits:
        return {"answer": "I don't know based on the documents.", "hits": []}
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return {"answer": None, "hits": hits}  # retrieval-only mode
    context = "\n\n".join(f"[{h['section']}]\n{h['text']}" for h in hits)
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
              "messages": [{"role": "user", "content": PROMPT.format(context=context, question=question)}],
              "temperature": 0},
        timeout=60,
    )
    r.raise_for_status()
    return {"answer": r.json()["choices"][0]["message"]["content"], "hits": hits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    q = ap.parse_args().question
    out = answer(q)
    if out["answer"]:
        print("Answer:", out["answer"], "\n")
    for h in out["hits"]:
        print(f"- [{h['section']}] distance={h['distance']:.3f}")
