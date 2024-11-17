from transformers import (
    DPRContextEncoder, 
    DPRContextEncoderTokenizer, 
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizer, 
    DPRReader, 
    DPRReaderTokenizer
)
import torch

# Initialize tokenizers and models
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-multiset-base")
reader = DPRReader.from_pretrained("facebook/dpr-reader-multiset-base")

# Define the query
query = "What are the benefits of solar energy?"

# Encode the query into embeddings
query_inputs = question_tokenizer(query, return_tensors="pt")
with torch.no_grad():
    query_embeddings = question_encoder(**query_inputs).pooler_output

# Define the documents (passages)
documents = [
    "Solar energy is a renewable source of power.",
    "It reduces electricity bills.",
    "It has low maintenance costs.",
    "Solar panels help combat climate change and reduce carbon footprint.",
    "Solar farms create new job opportunities in renewable energy sectors."
]

# Encode the documents into embeddings
doc_embeddings = []
for doc in documents:
    doc_inputs = context_tokenizer(doc, return_tensors="pt")
    with torch.no_grad():
        doc_embeddings.append(context_encoder(**doc_inputs).pooler_output)
doc_embeddings = torch.cat(doc_embeddings)

# Compute similarity scores
scores = torch.matmul(query_embeddings, doc_embeddings.T).squeeze()

# Rank documents based on their similarity scores
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# Print ranked documents
print("Query:", query)
print("\nRanked Documents:")
for doc, score in ranked_docs:
    print(f"Document: {doc}, Score: {score.item():.4f}")

# Extract passages for the reader
passages = [doc for doc, score in ranked_docs]

# Prepare inputs for the reader
inputs = reader_tokenizer(
    questions=query,
    titles=["Passage"] * len(passages),
    texts=passages,
    return_tensors="pt",
    padding=True,
    truncation=True
)

# Use the reader to extract the most relevant passage
with torch.no_grad():
    outputs = reader(**inputs)

# Extract the passage with the highest score
max_score_index = torch.argmax(outputs.relevance_logits)
most_relevant_passage = passages[max_score_index]

print("\nMost Relevant Passage:", most_relevant_passage)
