"""Embedding functions.

minilm  (default): all-MiniLM-L6-v2 through Chroma's local ONNX runtime. Real semantic embeddings.
                   The model (~80 MB) is downloaded on first use, so it needs internet access.
hashing           : offline fallback. Word and character n-grams hashed into a fixed-size vector.
                   No download, deterministic, but it matches wording, not meaning.
"""
from __future__ import annotations
import os
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

# Cosine-distance cut-off below which a hit counts as relevant. Different embedders
# produce different distance ranges, so each one has its own value.
MAX_DISTANCE = {"minilm": 0.65, "hashing": 0.90}  # minilm value is an untuned guess; hashing value chosen from the eval set


def embedder_name() -> str:
    return os.getenv("EMBEDDER", "minilm").lower()


class HashingEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.word = HashingVectorizer(n_features=n_features, analyzer="word", ngram_range=(1, 2),
                                      alternate_sign=False, norm="l2")
        self.char = HashingVectorizer(n_features=n_features, analyzer="char_wb", ngram_range=(3, 5),
                                      alternate_sign=False, norm="l2")

    def __call__(self, input: Documents) -> Embeddings:
        w = self.word.transform(input).toarray()
        c = self.char.transform(input).toarray()
        v = np.hstack([w, c])
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v.astype(np.float32).tolist()

    @staticmethod
    def name() -> str:
        return "hashing_ngrams"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashingEmbedding()


def get_embedding_function():
    if embedder_name() == "hashing":
        return HashingEmbedding()
    return embedding_functions.DefaultEmbeddingFunction()
