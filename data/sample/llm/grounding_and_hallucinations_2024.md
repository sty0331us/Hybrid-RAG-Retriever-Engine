# LLM Grounding and Hallucination Control (2024)

Large language models can produce fluent but incorrect statements.
In enterprise settings this is unacceptable for policies, pricing, and compliance.

## Grounding techniques

1. Retrieval-Augmented Generation with strict "answer only from context" prompts
2. Citation requirements tied to retrieved chunk ranks
3. Refusal behavior when context is insufficient
4. Temperature near zero for factual workloads

## Why retriever choice matters

If the retriever returns the wrong neighborhood of the corpus, the generator
will confidently invent bridging facts. Multi-query and self-query strategies
can improve coverage and constraint satisfaction respectively, while parent
document retrieval improves contextual completeness for long source materials.
