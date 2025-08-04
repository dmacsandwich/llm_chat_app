from sqlalchemy import text
from app.core.ports import HistoryRepoPort

class HistoryRepository(HistoryRepoPort):
    def __init__(self, engine):
        self.engine = engine

    def save(self, user_id: str, chat_history: list[dict]) -> None:
        sql = text("""
            INSERT INTO user_history (user_id, chat_history)
            VALUES (:user_id, :chat_history::jsonb)
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, {"user_id": user_id, "chat_history": json.dumps(chat_history)})

import json  # keep import bottom to emphasize only JSON payload converting
