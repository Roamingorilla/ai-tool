```markdown
# Architecture Overview

Components:
- Ingest: parsers for PDF / Word / HTML -> chunker -> embedding job
- Vector DB: Weaviate / Pinecone / Chroma for retrieval
- Embeddings service: OpenAI or open-source (sentence-transformers)
- LLM Orchestrator: LangChain-based chains/agents that create responses and call action plugins
- Connectors: HTTP chat endpoint, Slack bot, email webhook
- Actions: Plugin interface (idempotent), e.g., create_ticket, send_email, query_crm
- Auth & Security: JWT auth, role-based permissions, secrets manager
- Observability: structured logs, metrics, tracing, prompt auditing

Data flow:
1. Document ingested -> split -> embed -> store
2. User asks question -> embed query -> vector DB search -> pass top-k context to LLM with prompt template
3. LLM returns answer or an action plan -> if action required, orchestrator runs action plugin (with confirmations if needed)
4. Response returned and stored in audit log

Operational concerns:
- Rate limits for LLMs and connectors
- PII redaction and redaction policies
- Cost tracking per tenant / user
- Reproducible prompt templates for testing
```