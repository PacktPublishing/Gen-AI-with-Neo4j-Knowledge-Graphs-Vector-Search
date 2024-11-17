from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Move model to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example query and retrieved passages
query = "What are the benefits of solar energy?"
retrieved_passages = """
Solar energy is a renewable resource and reduces electricity bills.
It is environmentally friendly and helps combat climate change.
Solar panels require minimal maintenance and have a long lifespan.
"""

def generate_response(query, retrieved_passages):
    # Combine query and retrieved passages into a task-specific input
    input_text = f"Answer this question based on the provided context: {query} Context: {retrieved_passages}"

    # Tokenize the input with truncation and padding
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,  # Allow longer responses
            num_beams=3,     # Use beam search for better results
            early_stopping=True
        )

    # Decode the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate and print the response
response = generate_response(query, retrieved_passages)
print("Query:", query)
print("Retrieved Passages:", retrieved_passages)
print("Generated Response:", response)
