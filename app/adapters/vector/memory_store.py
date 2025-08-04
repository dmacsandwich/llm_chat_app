from typing import Sequence, List, Tuple
import math
from app.core.ports import VectorStorePort, RetrievalHit

def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    cos_sim = dot / (na*nb)
    return 1.0 - cos_sim  # distance-style (lower is better)

class InMemoryVectorStore(VectorStorePort):
    def __init__(self):
        self._items: list[Tuple[list[float], str]] = []

    def add(self, texts, embeddings):
        for t, e in zip(texts, embeddings):
            self._items.append((list(e), t))

    def search(self, embedding: Sequence[float], top_k: int = 5) -> List[RetrievalHit]:
        scored = [(t, _cosine_distance(embedding, e)) for e, t in self._items]
        scored.sort(key=lambda x: x[1])
        return [RetrievalHit(text=t, score=s) for t, s in scored[:top_k]]
