from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRReader,
    DPRReaderTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
import torch

# Initialize DPR tokenizers and encoders
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-multiset-base")
reader = DPRReader.from_pretrained("facebook/dpr-reader-multiset-base")

# Initialize T5 tokenizer and model for generation
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Define query and corpus
query = "What are the benefits of solar energy?"
documents = [
    "Solar energy is a renewable source of power.",
    "It reduces electricity bills.",
    "It has low maintenance costs.",
    "Solar panels help combat climate change and reduce carbon footprint.",
    "Solar farms create new job opportunities in renewable energy sectors."
]

# Encode the query
query_inputs = question_tokenizer(query, return_tensors="pt")
with torch.no_grad():
    query_embeddings = question_encoder(**query_inputs).pooler_output

# Encode the documents
doc_embeddings = []
for doc in documents:
    doc_inputs = context_tokenizer(doc, return_tensors="pt")
    with torch.no_grad():
        doc_embeddings.append(context_encoder(**doc_inputs).pooler_output)
doc_embeddings = torch.cat(doc_embeddings)

# Compute similarity scores and rank documents
scores = torch.matmul(query_embeddings, doc_embeddings.T).squeeze()
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# Print ranked documents
print("Query:", query)
print("\nRanked Documents:")
for doc, score in ranked_docs:
    print(f"Document: {doc}, Score: {score.item():.4f}")

# Extract top passages (e.g., top 3)
retrieved_docs = [doc for doc, score in ranked_docs[:3]]

# Define integration and generation function
def integrate_and_generate(query, retrieved_docs):
    """
    Integrates retrieved documents with the query and generates a synthesized response using T5.
    """
    # Combine query and retrieved documents into a single input
    input_text = f"Answer this question based on the following context: {query} Context: {' '.join(retrieved_docs)}"
    
    # Tokenize input for T5
    inputs = t5_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate a response
    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_length=100)
    
    # Decode and return the generated response
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate the response
response = integrate_and_generate(query, retrieved_docs)

# Print the final response
print("\nGenerated Response:", response)
