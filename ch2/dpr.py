from transformers import DPRContextEncoderTokenizer, DPRContextEncoder
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the tokenizer and model
tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')

# Example document database (unstructured)
documents = [
    "The IPL 2024 was a thrilling season with unexpected results.",
    "The invention of the printing press revolutionized communication in the 15th century.",
    "Neo4j is a graph database platform for connected data applications.",
    "ChatGPT is a large language model developed by OpenAI.",
    "Dense Passage Retrieval (DPR) is a state-of-the-art technique for information retrieval."
]

# Precompute embeddings for the document database
def encode_documents(documents):
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy()

document_embeddings = encode_documents(documents)

# Function to retrieve top documents based on a query
def retrieve_documents(query, num_results=3):
    # Encode the query
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model(**inputs).pooler_output.numpy()
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(query_embedding, document_embeddings).flatten()
    
    # Get top results
    top_indices = similarity_scores.argsort()[-num_results:][::-1]
    top_docs = [(documents[i], similarity_scores[i]) for i in top_indices]
    
    return top_docs

# Test the retrieval system
query = "What is Dense Passage Retrieval?"
top_results = retrieve_documents(query)

print("Query:", query)
print("\nTop Results:")
for doc, score in top_results:
    print(f"Score: {score:.4f}, Document: {doc}")
