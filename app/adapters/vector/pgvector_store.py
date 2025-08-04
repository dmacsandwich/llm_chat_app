from typing import Sequence, List
from sqlalchemy import text
from app.core.ports import VectorStorePort, RetrievalHit

def _vec_literal(v: Sequence[float]) -> str:
    # pgvector literal format: '[v1,v2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

class PGVectorStore(VectorStorePort):
    def __init__(self, engine, table: str = "documents"):
        self.engine = engine
        self.table = table

    def add(self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None:
        assert len(texts) == len(embeddings)
        sql = text(f"INSERT INTO {self.table} (embeddings, context) VALUES (:emb, :ctx)")
        with self.engine.begin() as conn:
            for t, e in zip(texts, embeddings):
                conn.execute(sql, {"emb": _vec_literal(e), "ctx": t})

    def search(self, embedding: Sequence[float], top_k: int = 5) -> List[RetrievalHit]:
        vec = _vec_literal(embedding)
        # cosine distance operator <=> ; lower is better
        sql = text(f"""
            SELECT context, (embeddings <=> :q) AS score
            FROM {self.table}
            ORDER BY embeddings <=> :q
            LIMIT :k
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"q": vec, "k": top_k}).all()
        return [RetrievalHit(text=r[0], score=float(r[1])) for r in rows]
