# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a parametric language model with a
non-parametric retriever over an external corpus. At query time the system
retrieves passages relevant to the question and conditions generation on them,
which grounds the answer in source documents and reduces hallucination.

## Hybrid retrieval

Dense retrieval embeds text into vectors and finds nearest neighbours by cosine
similarity, capturing semantic meaning. Lexical retrieval such as BM25 matches
exact terms. Hybrid retrieval fuses both — often with reciprocal rank fusion —
and then reranks the top candidates with a cross-encoder for precision.
