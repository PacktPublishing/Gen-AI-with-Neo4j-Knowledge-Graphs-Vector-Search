from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch

# Initialize DPR tokenizers and encoders
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Define the query and documents
query = "What are the benefits of solar energy?"
documents = [
    "Solar energy is a renewable source of power.",  # Related to solar energy
    "It reduces electricity bills.",  # Related to solar energy
    "It has low maintenance costs.",  # Related to solar energy
    "Solar panels help combat climate change.",  # Related to solar energy
    "Wind energy is another sustainable energy source.",  # Related to renewable energy, not specific to solar
    "Batteries are used to store energy for later use.",  # Related to energy, generic
    "Graph databases like Neo4j are used to model complex relationships.",  # Unrelated, technology-specific
    "Artificial intelligence is transforming industries worldwide.",  # Unrelated, generic technology
    "Electric cars are gaining popularity due to their low emissions.",  # Related to renewable energy applications
    "Water conservation is critical for sustainable development.",  # Unrelated, environmental
    "Machine learning models require large datasets for training.",  # Unrelated, AI/ML-specific
    "Solar energy farms are expanding rapidly in urban areas."  # Related to solar energy
]


# Encode the query into embeddings
query_inputs = question_tokenizer(query, return_tensors="pt")
with torch.no_grad():
    query_embeddings = question_encoder(**query_inputs).pooler_output

# Encode the documents into embeddings
doc_embeddings = []
for doc in documents:
    doc_inputs = context_tokenizer(doc, return_tensors="pt")
    with torch.no_grad():
        doc_embeddings.append(context_encoder(**doc_inputs).pooler_output)
doc_embeddings = torch.cat(doc_embeddings)

# Compute similarity scores (dot product)
scores = torch.matmul(query_embeddings, doc_embeddings.T).squeeze()

# Rank documents based on similarity scores
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# Print the ranked documents with their scores
print("Query:", query)
print("\nRanked Documents:")
for doc, score in ranked_docs:
    print(f"Document: {doc}, Score: {score.item():.4f}")
