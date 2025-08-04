# No environment variables used; adjust constants here.
AWS_REGION = "us-east-1"
BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
BEDROCK_LLM_MODEL_ID = "meta.llama3-1-8b-instruct-v1:0"
SECRETS_MANAGER_DB_SECRET_ID = "your/postgres/connection/secret"  # e.g., contains host, port, dbname, username, password

# pgvector dimension must match embedding dim (Titan v2 => 1024)
EMBED_DIM = 1024

# RAG defaults
TOP_K_DB = 5
TOP_K_MEMORY = 4
