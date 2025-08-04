from typing import Protocol, Sequence, Iterable
from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievalHit:
    text: str
    score: float  # lower is better for distances like cosine distance in pgvector

class EmbedderPort(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

class ChatLLMPort(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

class VectorStorePort(Protocol):
    def add(self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None: ...
    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[RetrievalHit]: ...

class HistoryRepoPort(Protocol):
    def save(self, user_id: str, chat_history: list[dict]) -> None: ...
