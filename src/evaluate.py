"""Measure retrieval quality on a small hand-written question set.

hit@1: the correct section is the top result. hit@3: it is among the top 3.
refusal: out-of-scope questions return no hit under the distance cut-off.
"""
from __future__ import annotations
import json
from pathlib import Path
from ask import retrieve
from embeddings import MAX_DISTANCE, embedder_name

TESTS = Path(__file__).resolve().parent.parent / "tests"


def run() -> dict:
    cases = json.loads((TESTS / "eval_set.json").read_text(encoding="utf-8"))
    oos = json.loads((TESTS / "out_of_scope.json").read_text(encoding="utf-8"))
    h1 = h3 = 0
    misses = []
    for c in cases:
        sections = [h["section"] for h in retrieve(c["q"], k=3)]
        h1 += sections[:1] == [c["section"]]
        h3 += c["section"] in sections
        if c["section"] not in sections[:1]:
            misses.append((c["q"], c["section"], sections[0]))
    cut = MAX_DISTANCE[embedder_name()]
    refused = sum(not [h for h in retrieve(q, k=3) if h["distance"] <= cut] for q in oos)
    return {"embedder": embedder_name(), "n": len(cases), "hit@1": h1, "hit@3": h3,
            "out_of_scope": len(oos), "refused": refused, "misses": misses}


if __name__ == "__main__":
    r = run()
    print(f"embedder={r['embedder']}  questions={r['n']}")
    print(f"hit@1: {r['hit@1']}/{r['n']}   hit@3: {r['hit@3']}/{r['n']}")
    print(f"out-of-scope refused: {r['refused']}/{r['out_of_scope']}")
    for q, want, got in r["misses"]:
        print(f"  miss: {q!r} expected={want!r} top1={got!r}")
