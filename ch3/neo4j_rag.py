from neo4j import GraphDatabase
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

# Define the connection credentials for the Neo4j database
uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
username = "neo4j"             # Replace with your Neo4j username
password = "password"          # Replace with your Neo4j password

# Create a driver instance to connect to the Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))

# Initialize the RAG tokenizer, model, and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

# Set the retriever for the model
model.set_retriever(retriever)

# Function to retrieve relevant data from the Neo4j knowledge graph
def get_relevant_data(prompt):
    """
    Fetch relevant data (plots) for movies that match the user's prompt.
    """
    query = f"""
    MATCH (m:Movie)-[:HAS_PLOT]->(p:Plot)
    WHERE m.title CONTAINS '{prompt}'
    RETURN m.title AS title, m.year AS year, p.description AS plot
    """
    with driver.session() as session:
        result = session.run(query)
        records = [
            {
                "title": record["title"],
                "year": record["year"],
                "plot": record["plot"],
            }
            for record in result if record["plot"] is not None
        ]
        print(f"Retrieved Records: {records}")  # Debugging line
        return records

# Function to generate a response using the RAG model
def generate_response(prompt):
    """
    Combine the user's prompt with relevant data from the graph
    and generate a focused, non-repetitive response using the RAG model.
    """
    relevant_data = get_relevant_data(prompt)

    if not relevant_data:
        return "No relevant data found for the given prompt."

    # Combine dictionaries in relevant_data into a single string
    combined_input = f"Provide detailed information about: {prompt}. " + " ".join(
        [f"{data['title']} ({data['year']}): {data['plot']}" for data in relevant_data]
    )

    print(f"Combined Input: {combined_input}")

    if not combined_input.strip():
        return "No relevant data to process for this prompt."

    # Tokenize the combined input with truncation
    max_input_length = 512 - 50  # Leave space for output
    tokenized_input = tokenizer(combined_input, truncation=True, max_length=max_input_length, return_tensors="pt")

    # Generate response with tuned parameters
    outputs = model.generate(
        **tokenized_input,
        max_length=150,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    # Decode the response with improved formatting
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response


# Example prompt provided within the script
prompt = "The Matrix"  # Replace with the movie title you want to query
response = generate_response(prompt)

# Print the AI-generated response
print(f"Prompt: {prompt}\nResponse: {response}")

# Close the database driver
driver.close()
