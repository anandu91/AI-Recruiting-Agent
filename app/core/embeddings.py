# app/core/embeddings.py
from __future__ import annotations

import os
import math
import hashlib
from typing import Any, Iterable, List, Optional

# -------------------- Config --------------------
_DEFAULT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
_DISABLE_ST = os.getenv("OPENAI_DISABLE_ST", "").strip().lower() in {"1", "true", "yes"}

# Fallback dimensionality (must be even; we use sin/cos pairs)
_FALLBACK_DIM = 32

# -------------------- Lazy backend load --------------------
_embedder: Optional[Any] = None
_backend: str = "unknown"  # "st" or "fallback"

def _try_load_st() -> Optional[Any]:
    """Try to load SentenceTransformer backend."""
    if _DISABLE_ST:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(_EMBED_MODEL_NAME)
    except Exception:
        return None

def _ensure_backend() -> None:
    global _embedder, _backend
    if _embedder is not None:
        return
    st = _try_load_st()
    if st is not None:
        _embedder = st
        _backend = "st"
    else:
        _embedder = "_fallback"
        _backend = "fallback"

# -------------------- Utils --------------------
def _l2_normalize(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / s for v in vec]

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    # Keep case for ST model; fallback is case-insensitive (hash of lower)
    return s

# -------------------- Deterministic 32-D fallback --------------------
def _fallback_embed_one(text: str, dim: int = _FALLBACK_DIM) -> List[float]:
    """
    Deterministic, dependency-free embedding:
    - Lowercase the text
    - For each i in [0..dim/2), compute a SHA-256 of (text || ":" || i)
    - Map hash -> angle in [0, 2π), produce [cos(angle), sin(angle)] pair
    - L2 normalize the final vector
    """
    t = (text or "").strip().lower()
    if dim % 2 != 0:
        dim += 1
    half = dim // 2
    out: List[float] = []
    for i in range(half):
        h = hashlib.sha256(f"{t}:{i}".encode("utf-8")).digest()
        # Use first 8 bytes as unsigned integer; map to [0, 2π)
        val = int.from_bytes(h[:8], "big", signed=False)
        angle = (val % 10_000_000) / 10_000_000.0 * (2.0 * math.pi)
        out.append(math.cos(angle))
        out.append(math.sin(angle))
    return _l2_normalize(out)

# -------------------- Public API --------------------
def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """
    Returns L2-normalized embeddings for the input texts.
    Primary: sentence-transformers (if available & not disabled).
    Fallback: deterministic 32-D hash-based sin/cos vectors.
    """
    _ensure_backend()
    texts = [ _clean_text(t) for t in (list(texts) if texts is not None else []) ]
    if not texts:
        return []

    if _backend == "st":
        # sentence-transformers path (already normalized within encode(normalize_embeddings=True))
        try:
            embs = _embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            # ensure python lists
            return [list(map(float, v)) for v in embs]
        except Exception:
            # If runtime failure occurs, fall back deterministically
            return [_fallback_embed_one(t) for t in texts]

    # deterministic fallback path
    return [_fallback_embed_one(t) for t in texts]

def embed_one(text: str) -> List[float]:
    """Convenience wrapper for a single string."""
    res = embed_texts([text])
    return res[0] if res else []
