import os
import time
from neo4j_haystack import Neo4jDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever
from haystack.utils.auth import Secret
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Neo4j Document Store from Haystack
def initialize_neo4j_document_store():
    document_store = Neo4jDocumentStore(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index="movies_embeddings"  # Custom index name for movie embeddings
    )
    return document_store

# Create or drop the vector index in Neo4j AuraDB
def create_or_reset_vector_index():
    with driver.session() as session:
        try:
            # Drop the existing vector index if it exists
            session.run("DROP INDEX IF EXISTS overview_embeddings")
            print("Old index dropped")
        except:
            print("No index to drop")

        # Create a new vector index on the embedding property
        print("Creating new vector index")
        query_index = """
        CREATE VECTOR INDEX `overview_embeddings` 
        FOR (m:Movie) ON (m.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,  
            `vector.similarity_function`: 'cosine'}}
        """    
        session.run(query_index)
        print("Vector index created successfully")

# Initialize Haystack Dense Retriever for vector search
def initialize_dense_retriever(document_store, embedder):
    retriever = Neo4jEmbeddingRetriever(
        document_store=document_store,
        embedding_model=embedder
    )
    return retriever

# Perform vector search using Haystack
def perform_vector_search(retriever, query):
    # Use the retriever to find movies with similar embeddings
    results = retriever.retrieve(query)
    
    # Output the results
    for result in results:
        print(f"Movie ID: {result.meta['tmdbId']}, Overview: {result.content}")

# Main function to orchestrate the entire process
def main():
    # Step 1: Initialize Neo4j Document Store and OpenAI Embedder
    document_store = initialize_neo4j_document_store()
    
    # Since embeddings are already generated in the previous section, we only initialize the embedder for vector search
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )

    # Step 2: Create or reset vector index in Neo4j AuraDB
    create_or_reset_vector_index()

    # Step 3: Initialize dense retriever for vector search
    retriever = initialize_dense_retriever(document_store, embedder)

    # Step 4: Perform a vector search with a sample query
    query = "A hero must save the world from destruction"  # Replace with a movie plot or custom query
    perform_vector_search(retriever, query)

if __name__ == "__main__":
    main()
