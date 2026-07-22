# Text embeddings

An embedding model maps text to a fixed-length vector so that semantically
similar texts land close together in the vector space. Sentence-transformer
models such as the BGE family are trained for retrieval: a query and a relevant
passage are pushed together, while unrelated pairs are pushed apart.

Normalizing embeddings to unit length makes cosine similarity equal to a dot
product, which is what most vector databases index. Retrieval models are often
asymmetric — the query is prefixed with a short instruction while passages are
left bare — which measurably improves search quality.
