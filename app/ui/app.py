import streamlit as st
from uuid import uuid4

# -------- config --------
from app.config.settings import (
    AWS_REGION,
    BEDROCK_EMBED_MODEL_ID,
    BEDROCK_LLM_MODEL_ID,
    SECRETS_MANAGER_DB_SECRET_ID,
    EMBED_DIM,
    TOP_K_DB,
    TOP_K_MEMORY,
)

# -------- adapters --------
from app.adapters.secrets.aws_secrets import get_db_secret
from app.adapters.db.postgres_repository import make_engine
from app.adapters.db.history_repository import HistoryRepository
from app.adapters.vector.pgvector_store import PGVectorStore
from app.adapters.vector.memory_store import InMemoryVectorStore
from app.adapters.bedrock.embeddings_bedrock import TitanEmbedder
from app.adapters.bedrock.llm_bedrock import LlamaChatLLM

# -------- domain / infra --------
from app.infrastructure.db_init import init_db
from app.domain.rag_service import RAGService
from app.domain.chat_service import ChatService

# -------------------------------------------------
st.set_page_config(page_title="Bedrock RAG Chat", page_icon="💬")

# ---------- boot singletons ----------
if "booted" not in st.session_state:
    # DB + tables
    db_secret = get_db_secret(SECRETS_MANAGER_DB_SECRET_ID, AWS_REGION)
    engine = make_engine(db_secret)
    init_db(engine, embed_dim=EMBED_DIM)

    # adapters
    embedder = TitanEmbedder(AWS_REGION, BEDROCK_EMBED_MODEL_ID)
    llm = LlamaChatLLM(AWS_REGION, BEDROCK_LLM_MODEL_ID)
    db_store = PGVectorStore(engine)
    mem_store = InMemoryVectorStore()
    history_repo = HistoryRepository(engine)

    # services
    rag = RAGService(embedder, db_store, mem_store, TOP_K_DB, TOP_K_MEMORY)
    chat_service = ChatService(llm, rag)

    # session values
    st.session_state.update(
        {
            "chat_service": chat_service,
            "history_repo": history_repo,
            "engine": engine,
            "user_id": "demo-user",
            "history": [],
            "conversation_id": None,
            "booted": True,
        }
    )

# ------------------------------------------------------------------
#  Sidebar – conversation list & new chat
# ------------------------------------------------------------------
repo = st.session_state["history_repo"]
uid = st.session_state["user_id"]

with st.sidebar:
    st.subheader("💬 Your Chats")

    if st.button("➕ New chat"):
        cid = repo.start_conversation(uid)
        st.session_state.update(
            {
                "conversation_id": cid,
                "history": [],
            }
        )
        st.session_state["chat_service"].rag.mem_store = InMemoryVectorStore()
        st.rerun()

    chats = repo.list_for_user(uid, limit=25)
    if chats:
        labels = [f"{c['title']} · {c['ts'].strftime('%m/%d %H:%M')}" for c in chats]
        idx = labels.index(
            next(
                (l for l, c in zip(labels, chats) if c["conversation_id"] == st.session_state.get("conversation_id")),
                labels[0],
            )
        )
        sel = st.selectbox("Recent conversations", labels, index=idx)
        chosen = chats[labels.index(sel)]
        if chosen["conversation_id"] != st.session_state.get("conversation_id"):
            st.session_state["conversation_id"] = chosen["conversation_id"]
            st.session_state["history"] = repo.load(chosen["conversation_id"])
            st.session_state["chat_service"].rag.mem_store = InMemoryVectorStore()
            # Seed memory with last turns
            st.session_state["chat_service"].rag.add_to_memory(
                [
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state["history"][-8:]
                ]
            )
            st.rerun()
    else:
        st.info("No previous chats.")

# ------------------------------------------------------------------
#  Main chat area
# ------------------------------------------------------------------
st.title("💬 Bedrock Llama 8B RAG Chat")

for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask anything...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        answer, updated_hist = st.session_state["chat_service"].answer(
            st.session_state["history"], prompt
        )

    # persist & update session
    cid = st.session_state.get("conversation_id")
    cid = repo.save(uid, updated_hist, conversation_id=cid, title=None)
    st.session_state.update({"conversation_id": cid, "history": updated_hist})

    with st.chat_message("assistant"):
        st.write(answer)
