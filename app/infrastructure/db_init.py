from sqlalchemy import text

def init_db(engine, embed_dim: int) -> None:
    """
    Create required tables & indexes if they don't exist yet.
    """
    stmts = [
        # pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector;",

        # documents: two columns only (spec) + cosine IVF index
        f"""CREATE TABLE IF NOT EXISTS documents (
                embeddings vector({embed_dim}) NOT NULL,
                context    text            NOT NULL
           );""",
        """CREATE INDEX IF NOT EXISTS documents_embeddings_cosine_idx
                 ON documents
              USING ivfflat (embeddings vector_cosine_ops)
               WITH (lists = 100);""",

        # per-chat history, keyed by conversation_id
        """CREATE TABLE IF NOT EXISTS user_history (
                conversation_id uuid PRIMARY KEY,
                user_id   text NOT NULL,
                title     text,
                chat_history jsonb NOT NULL,
                ts        timestamptz NOT NULL DEFAULT now()
           );""",
        "CREATE INDEX IF NOT EXISTS user_history_user_idx "
        "          ON user_history(user_id, ts DESC);",
    ]

    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
