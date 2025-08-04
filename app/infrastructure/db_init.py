from sqlalchemy import text

def init_db(engine, embed_dim: int):
    stmts = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"""CREATE TABLE IF NOT EXISTS documents (
                embeddings vector({embed_dim}) NOT NULL,
                context    text NOT NULL
            );""",
        """CREATE TABLE IF NOT EXISTS user_history (
                user_id text NOT NULL,
                chat_history jsonb NOT NULL,
                ts timestamptz NOT NULL DEFAULT now()
            );""",
        # Optional: cosine index to speed search once you have data
        "CREATE INDEX IF NOT EXISTS documents_embeddings_cosine_idx ON documents USING ivfflat (embeddings vector_cosine_ops) WITH (lists = 100);"
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
