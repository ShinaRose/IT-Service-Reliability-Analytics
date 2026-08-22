"""Sentence embeddings (all-MiniLM-L6-v2) with a DuckDB content-hash cache.

Falls back to a deterministic hashing-vectorizer embedding if sentence-transformers
or its model weights aren't available (e.g. offline dev), so the rest of the pipeline
keeps working -- but the default, real path is sentence-transformers.
"""
from __future__ import annotations

import hashlib

import numpy as np

from relplatform.config import EMBEDDING_MODEL

_model = None
_EMBED_DIM = 384


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:  # pragma: no cover - offline fallback
            print(f"[embeddings] sentence-transformers unavailable ({e}); using hashing fallback")
            _model = "fallback"
    return _model


def _fallback_embed(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(n_features=_EMBED_DIM, norm="l2", alternate_sign=False)
    return vec.transform(texts).toarray().astype(np.float32)


def embed_texts(con, texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Returns (n, dim) float32 array, using the DuckDB `embedding_cache` table keyed by
    sha256(model || text)."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            content_hash VARCHAR PRIMARY KEY,
            model        VARCHAR,
            dim          INTEGER,
            vector       BLOB
        )
        """
    )
    hashes = [_hash(EMBEDDING_MODEL + "|" + t) for t in texts]
    existing = {}
    if hashes:
        placeholders = ",".join("?" for _ in set(hashes))
        rows = con.execute(
            f"SELECT content_hash, vector, dim FROM embedding_cache WHERE content_hash IN ({placeholders})",
            list(set(hashes)),
        ).fetchall()
        for h, blob, dim in rows:
            existing[h] = np.frombuffer(blob, dtype=np.float32, count=dim)

    missing_idx = [i for i, h in enumerate(hashes) if h not in existing]
    if missing_idx:
        model = _get_model()
        missing_texts = [texts[i] for i in missing_idx]
        if model == "fallback":
            new_vecs = _fallback_embed(missing_texts)
        else:
            new_vecs = model.encode(
                missing_texts, batch_size=batch_size, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype(np.float32)
        rows_to_insert = []
        for i, vec in zip(missing_idx, new_vecs):
            h = hashes[i]
            existing[h] = vec
            rows_to_insert.append((h, EMBEDDING_MODEL, int(vec.shape[0]), vec.tobytes()))
        con.executemany(
            "INSERT OR REPLACE INTO embedding_cache (content_hash, model, dim, vector) VALUES (?,?,?,?)",
            rows_to_insert,
        )

    dim = len(next(iter(existing.values())))
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, h in enumerate(hashes):
        out[i] = existing[h]
    return out
