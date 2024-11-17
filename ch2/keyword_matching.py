from rank_bm25 import BM25Okapi

# Example corpus of documents
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog quickly.",
    "The dog is quick and the fox is brown.",
    "Cats are smarter than dogs.",
    "Artificial intelligence is transforming the world."
]

# Tokenize the corpus (split each document into words)
tokenized_corpus = [doc.split() for doc in corpus]

# Initialize BM25 with the tokenized corpus
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# Tokenize the query
query = "quick fox"
tokenized_query = query.split()

# Compute BM25 scores for each document
scores = bm25.get_scores(tokenized_query)

# Rank the documents based on their scores
ranked_docs = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)

# Print the ranked documents with their scores
print("Query:", query)
print("\nRanked Documents:")
for doc, score in ranked_docs:
    print(f"Document: {doc}, Score: {score:.4f}")
