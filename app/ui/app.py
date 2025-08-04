import streamlit as st

from app.config.settings import (
    AWS_REGION, BEDROCK_EMBED_MODEL_ID, BEDROCK_LLM_MODEL_ID,
    SECRETS_MANAGER_DB_SECRET_ID, EMBED_DIM, TOP_K_DB, TOP_K_MEMORY
)
from app.adapters.secrets.aws_secrets import get_db_secret
from app.adapters.db.postgres_repository import make_engine
from app.adapters.db.history_repository import HistoryRepository
from app.adapters.vector.pgvector_store import PGVectorStore
from app.adapters.vector.memory_store import InMemoryVectorStore
from app.adapters.bedrock.embeddings_bedrock import TitanEmbedder
from app.adapters.bedrock.llm_bedrock import LlamaChatLLM
from app.infrastructure.db_init import init_db
from app.domain.rag_service import RAGService
from app.domain.chat_service import ChatService

st.set_page_config(page_title="Bedrock RAG Chat (pgvector + Streamlit)", page_icon="💬")

# Boot once per session
if "booted" not in st.session_state:
    # Secrets & DB
    db_secret = get_db_secret(SECRETS_MANAGER_DB_SECRET_ID, AWS_REGION)
    engine = make_engine(db_secret)
    init_db(engine, embed_dim=EMBED_DIM)

    # Adapters
    embedder = TitanEmbedder(region=AWS_REGION, model_id=BEDROCK_EMBED_MODEL_ID)
    llm = LlamaChatLLM(region=AWS_REGION, model_id=BEDROCK_LLM_MODEL_ID)

    db_store = PGVectorStore(engine)
    mem_store = InMemoryVectorStore()

    history_repo = HistoryRepository(engine)

    # Domain services
    rag = RAGService(embedder, db_store, mem_store, TOP_K_DB, TOP_K_MEMORY)
    chat = ChatService(llm, rag, history_repo)

    # Keep handles in session
    st.session_state["chat_service"] = chat
    st.session_state["history"] = []
    st.session_state["user_id"] = "demo-user"  # replace with your auth/user identity if available
    st.session_state["booted"] = True

st.title("💬 RAG Chat — Bedrock + Llama 3.1 8B + Titan v2 + pgvector")

# Sidebar: simple loader to seed vector store (optional)
with st.sidebar:
    st.header("Corpus Loader (optional)")
    seed_text = st.text_area("Add context (one chunk per line)", height=150, placeholder="Paste a few lines...")
    if st.button("Embed + store in Postgres"):
        if seed_text.strip():
            lines = [ln.strip() for ln in seed_text.splitlines() if ln.strip()]
            embeds = st.session_state["chat_service"].rag.embedder.embed_batch(lines)
            st.session_state["chat_service"].rag.db_store.add(lines, embeds)
            st.success(f"Inserted {len(lines)} rows into documents.")
    st.caption("DB table `documents(embeddings, context)`; cosine index used for search.")

# Chat UI
for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask me anything...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        answer, updated = st.session_state["chat_service"].answer(
            user_id=st.session_state["user_id"],
            chat_history=st.session_state["history"],
            user_query=prompt
        )

    st.session_state["history"] = updated
    with st.chat_message("assistant"):
        st.write(answer)
