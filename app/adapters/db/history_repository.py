from __future__ import annotations

from uuid import uuid4
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from app.core.ports import HistoryRepoPort


class HistoryRepository(HistoryRepoPort):
    """
    CRUD helper for the user_history table.
    """

    def __init__(self, engine):
        self.engine = engine

    # ---------- helpers ----------

    def _insert_empty(self, user_id: str, title: str | None) -> str:
        cid = str(uuid4())
        sql = text(
            "INSERT INTO user_history (conversation_id, user_id, title, chat_history) "
            "VALUES (:cid, :uid, :title, :hist)"
        ).bindparams(bindparam("hist", type_=JSONB))
        with self.engine.begin() as conn:
            conn.execute(sql, {"cid": cid, "uid": user_id, "title": title, "hist": []})
        return cid

    # ---------- public API ----------

    def start_conversation(self, user_id: str, title: str | None = None) -> str:
        """
        Create a blank chat row and return its UUID.
        """
        return self._insert_empty(user_id, title)

    def load(self, conversation_id: str) -> list[dict]:
        sql = text("SELECT chat_history FROM user_history WHERE conversation_id = :cid")
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"cid": conversation_id}).first()
        return row[0] if row else []

    def save(
        self,
        user_id: str,
        chat_history: list[dict],
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> str:
        """
        Update (or create) a conversation. Returns its id.
        """
        if conversation_id is None:
            conversation_id = self._insert_empty(user_id, title)

        sql = text(
            "UPDATE user_history "
            "   SET chat_history = :hist, ts = now() "
            " WHERE conversation_id = :cid"
        ).bindparams(bindparam("hist", type_=JSONB))
        with self.engine.begin() as conn:
            conn.execute(sql, {"cid": conversation_id, "hist": chat_history})
        return conversation_id

    def list_for_user(self, user_id: str, limit: int = 25) -> list[dict]:
        """
        Return recent conversations for sidebar selector.
        """
        sql = text(
            "SELECT conversation_id, coalesce(title, '') AS title, ts, chat_history "
            "  FROM user_history "
            " WHERE user_id = :uid "
            " ORDER BY ts DESC "
            " LIMIT :lim"
        )
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"uid": user_id, "lim": limit}).all()

        out = []
        for cid, title, ts, hist in rows:
            if not title:
                # derive a quick title from first user utterance
                first_user = next(
                    (m["content"] for m in hist if m.get("role") == "user"), ""
                )
                title = first_user[:80] or "New chat"
            out.append({"conversation_id": cid, "title": title, "ts": ts})
        return out
