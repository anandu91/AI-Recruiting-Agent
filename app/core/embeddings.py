from typing import Any, Iterable, List
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"
_embedder: Any = None

def get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    texts = list(texts)
    if not texts:
        return []
    emb = get_embedder().encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return emb.tolist()
