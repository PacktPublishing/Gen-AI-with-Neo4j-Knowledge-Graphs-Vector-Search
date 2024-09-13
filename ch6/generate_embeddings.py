import os
import openai
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret
from neo4j import GraphDatabase
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Haystack with OpenAI for text embeddings
def initialize_haystack():
    # Initialize document store (In-memory for now, but you can configure other stores)
    document_store = InMemoryDocumentStore()

    # Initialize OpenAITextEmbedder to generate text embeddings
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )
    return embedder

# Retrieve movie plots and titles from Neo4j
def retrieve_movie_plots():
    # The query retrieves the "title", "overview", and "tmdbId" properties of each Movie node
    query = "MATCH (m:Movie) WHERE m.embedding IS NULL RETURN m.tmdbId AS tmdbId, m.title AS title, m.overview AS overview"
    with driver.session() as session:
        results = session.run(query)
        # Each movie's title, plot (overview), and ID are retrieved and stored in the movies list
        movies = [{"tmdbId": row["tmdbId"], "title": row["title"], "overview": row["overview"]} for row in results]
    return movies

# Generate embeddings for movie plots using Haystack and store them immediately in Neo4j
def generate_and_store_embeddings(embedder, movies):
    for movie in movies:
        title = movie.get("title", "Unknown Title")  # Fetch the movie title
        overview = str(movie.get("overview", ""))  # Ensure the overview is a string, use empty string as default
        
        print(f"Generating embedding for movie: {title}")
        print(f"Overview for {title} movie: {overview}")

        # Check if the overview is not empty
        if overview.strip() == "":
            print(f"No overview available for movie: {title}. Skipping embedding generation.")
            continue
        
        try:
            # Generate embedding for the current overview (pass overview as a string to the embedder)
            embedding_result = embedder.run(overview)  # Pass overview as a string
            embedding = embedding_result.get("embedding", None)  # Safely access the embedding from the result
            
            if embedding:
                # Store the embedding in Neo4j immediately
                tmdbId = movie["tmdbId"]
                store_embedding_in_neo4j(tmdbId, embedding)
            else:
                print(f"Failed to generate embedding for movie: {title}")
        except Exception as e:
            print(f"Error generating embedding for movie {title}: {e}")

# Store the embedding in Neo4j
def store_embedding_in_neo4j(tmdbId, embedding):
    query = """
    MATCH (m:Movie {tmdbId: $tmdbId})
    SET m.embedding = $embedding
    """
    with driver.session() as session:
        # Directly store the embedding (no need for .tolist())
        session.run(query, tmdbId=tmdbId, embedding=embedding)
    print(f"Embedding for movie {tmdbId} successfully stored in Neo4j.")

# Verify embeddings stored in Neo4j
def verify_embeddings():
    query = "MATCH (m:Movie) WHERE m.embedding IS NOT NULL RETURN m.title, m.embedding LIMIT 10"
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            print(f"Movie: {record['m.title']}, Embedding: {np.array(record['m.embedding'])[:5]}...")  # Print only first 5 values of embedding for brevity

# Main function to orchestrate the entire process
def main():
    # Step 1: Initialize Haystack with OpenAI embeddings retriever
    embedder = initialize_haystack()

    # Step 2: Retrieve movie plots from Neo4j
    movies = retrieve_movie_plots()
    if not movies:
        print("No movies found in the Neo4j database.")
        return

    # Step 3: Generate embeddings for movie plots and store them immediately
    generate_and_store_embeddings(embedder, movies)

    # Step 4: Verify that embeddings are stored in Neo4j
    verify_embeddings()

if __name__ == "__main__":
    main()