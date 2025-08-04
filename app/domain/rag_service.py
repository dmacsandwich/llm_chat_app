from typing import List
from app.core.ports import EmbedderPort, VectorStorePort, RetrievalHit

class RAGService:
    def __init__(self, embedder: EmbedderPort, db_store: VectorStorePort, mem_store: VectorStorePort,
                 top_k_db: int = 5, top_k_mem: int = 4):
        self.embedder = embedder
        self.db_store = db_store
        self.mem_store = mem_store
        self.top_k_db = top_k_db
        self.top_k_mem = top_k_mem

    def retrieve(self, query: str) -> List[RetrievalHit]:
        q_emb = self.embedder.embed(query)
        hits_db = self.db_store.search(q_emb, top_k=self.top_k_db)
        hits_mem = self.mem_store.search(q_emb, top_k=self.top_k_mem)
        # simple merge: take all and sort by score
        merged = sorted(hits_db + hits_mem, key=lambda h: h.score)
        return merged

    def add_to_memory(self, texts: list[str]) -> None:
        if not texts:
            return
        embeddings = self.embedder.embed_batch(texts)
        self.mem_store.add(texts, embeddings)
