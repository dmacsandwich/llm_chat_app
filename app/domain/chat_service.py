from typing import List
from app.core.ports import ChatLLMPort, HistoryRepoPort
from app.domain.rag_service import RAGService

SYSTEM_BASE = (
    "You are a helpful assistant. Use the provided context if relevant; "
    "otherwise answer from general knowledge. Be concise."
)

class ChatService:
    def __init__(self, llm: ChatLLMPort, rag: RAGService, history_repo: HistoryRepoPort):
        self.llm = llm
        self.rag = rag
        self.history_repo = history_repo

    def answer(self, user_id: str, chat_history: List[dict], user_query: str) -> tuple[str, List[dict]]:
        # Retrieve context
        hits = self.rag.retrieve(user_query)
        context = "\n\n".join(h.text for h in hits)

        # Compose Bedrock Converse-style messages
        messages = [
            {"role": "system", "content": [{"text": SYSTEM_BASE}]},
        ]
        if context.strip():
            messages.append({"role": "system", "content": [{"text": f"Context:\n{context}"}]})
        # add prior turns (short form)
        for m in chat_history[-10:]:
            messages.append({"role": m["role"], "content": [{"text": m["content"]}]})
        messages.append({"role": "user", "content": [{"text": user_query}]})

        answer = self.llm.chat(messages)

        # Update in-memory RAG with fresh content to keep chat “up”
        self.rag.add_to_memory([f"USER: {user_query}", f"ASSISTANT: {answer}"])

        # Persist history (append then save)
        updated = chat_history + [{"role": "user", "content": user_query},
                                  {"role": "assistant", "content": answer}]
        self.history_repo.save(user_id, updated)
        return answer, updated
