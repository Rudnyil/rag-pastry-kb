# rag-pastry-kb

A small Retrieval-Augmented Generation (RAG) project in Python: ask questions about a document set and get answers grounded in it, with a vector database underneath.

The demo knowledge base is the policy document of a **fictional** pastry shop ("Sweet Dream", all data invented). It stands in for the kind of internal documents a small business has: lead times, prices, delivery, allergens, refunds.

## What it does

1. **Chunking** – `src/chunking.py` splits Markdown by `##` sections (long sections are split with overlap), keeping the section name as metadata.
2. **Indexing** – `src/ingest.py` embeds the chunks and upserts them into a persistent **Chroma** vector database (cosine distance). Re-running it does not create duplicates.
3. **Retrieval** – `src/ask.py` embeds the question and returns the top 3 chunks. If nothing is closer than a distance threshold, it answers "I don't know based on the documents."
4. **Answering (optional)** – with an `OPENROUTER_API_KEY` the retrieved chunks are sent to an LLM with a context-only prompt. Without a key the tool prints the retrieved passages (retrieval-only mode).
5. **Evaluation** – `src/evaluate.py` measures retrieval on 16 hand-written questions (`tests/eval_set.json`) and 4 out-of-scope questions.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python ingest.py
python ask.py "Do you deliver on Sunday?"
python evaluate.py
```

Embedders (`EMBEDDER` environment variable):

| Value | What it is | Needs internet |
|---|---|---|
| `minilm` (default) | all-MiniLM-L6-v2 via Chroma's local ONNX runtime, real semantic embeddings | yes, ~80 MB model download on first run |
| `hashing` | word and character n-grams hashed into vectors; matches wording, not meaning | no (`pip install scikit-learn`) |

Each embedder uses its own collection, so vectors are never mixed.

## Measured results (and what is not measured)

Run in a sandbox without access to the model host, so **only the `hashing` embedder was run**:

| Embedder | hit@1 | hit@3 | Out-of-scope refused |
|---|---|---|---|
| hashing | 12/16 | 15/16 | 1/4 |

- hit@k = the correct section appears in the top k results. The eval set is small and written by the author, so treat the numbers as a smoke test, not a benchmark.
- The hashing embedder cannot separate relevant from irrelevant questions reliably (distances overlap), which is why only 1 of 4 out-of-scope questions is refused. This is expected for a lexical baseline.
- **The default `minilm` embedder has not been run or evaluated yet**, and its distance threshold (0.65) is an untuned guess. Run `python evaluate.py` locally and adjust `MAX_DISTANCE` in `src/embeddings.py` from the observed distances.
- The LLM answering step has not been tested against a live API.

## Tests

```bash
EMBEDDER=hashing pytest -q   # 3 tests: chunking, idempotent ingest, retrieval
```

## Limits

- Demo data only; one small Markdown file.
- No re-ranking, no hybrid search, no conversation memory.
- PDF loading (`pypdf` is in requirements) is not wired in yet.

## Next steps

- Evaluate and tune with MiniLM; add more questions.
- Add PDF ingestion and a hybrid (keyword + vector) retriever.
