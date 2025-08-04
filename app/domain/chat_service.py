from typing import List
from app.core.ports import ChatLLMPort
from app.domain.rag_service import RAGService

SYSTEM_BASE = (
    "You are a helpful assistant. Use provided context if relevant; "
    "otherwise answer from general knowledge."
)


class ChatService:
    """
    Pure domain logic: create prompt, call LLM, update in-memory RAG.
    Persistence happens in the UI layer.
    """

    def __init__(self, llm: ChatLLMPort, rag: RAGService):
        self.llm = llm
        self.rag = rag

    def answer(self, chat_history: List[dict], user_query: str) -> tuple[str, List[dict]]:
        # --- retrieve ---
        hits = self.rag.retrieve(user_query)
        context = "\n\n".join(h.text for h in hits).strip()

        # --- craft messages ---
        system_blocks = [{"text": SYSTEM_BASE}]
        if context:
            system_blocks.append({"text": f"Context:\n{context}"})

        messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in chat_history[-10:]  # last N
            if m["role"] in ("user", "assistant")
        ]
        messages.append({"role": "user", "content": [{"text": user_query}]})

        # --- LLM call ---
        answer_text = self.llm.chat(
            [{"role": "system", "content": system_blocks}, *messages]
        )

        # --- update memory store for short-term recall ---
        self.rag.add_to_memory([f"USER: {user_query}", f"ASSISTANT: {answer_text}"])

        # --- return updated history ---
        updated_hist = chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer_text},
        ]
        return answer_text, updated_hist
