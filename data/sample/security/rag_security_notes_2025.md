# Security Notes for RAG Deployments (2025)

RAG systems introduce unique security considerations beyond classic web apps.

## Data handling

- Do not embed secrets into the vector index.
- Redact PII before chunking when policies require it.
- Separate collections by tenant to avoid cross-customer retrieval leakage.

## Prompt injection via documents

Untrusted documents can contain instructions that attempt to override system
prompts. Mitigations include:
- treating retrieved text as untrusted data
- disallowing tool execution based solely on retrieved content
- output filters for exfiltration patterns

## API keys and configuration

Load credentials from environment variables or a secret manager.
Never commit `.env` files. Rotate keys when engineers leave the team.
