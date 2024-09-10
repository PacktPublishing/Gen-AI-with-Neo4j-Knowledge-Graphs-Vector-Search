import os
from neo4j_haystack import Neo4jDocumentStore
from neo4j_haystack import Neo4jDynamicDocumentRetriever
from neo4j_haystack import Neo4jClientConfig
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Create or drop the vector index in Neo4j AuraDB
def create_or_reset_vector_index():
    with driver.session() as session:
        try:
            # Drop the existing vector index if it exists
            session.run("DROP INDEX overview_embeddings IF EXISTS ")
            print("Old index dropped")
        except:
            print("No index to drop")

        # Create a new vector index on the embedding property
        print("Creating new vector index")
        query_index = """
        CREATE VECTOR INDEX overview_embeddings IF NOT EXISTS
        FOR (m:Movie) ON (m.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 384,  
            `vector.similarity_function`: 'cosine'}}
        """    
        session.run(query_index)
        print("Vector index created successfully")

client_config = Neo4jClientConfig(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database="neo4j",
)


# Perform vector search using Haystack
def perform_vector_search(query):

    document_store = Neo4jDocumentStore(
        client_config=client_config,
        index="overview_embeddings",  # The name of the Vector Index in Neo4j
        node_label="Movie",  # Providing a label to Neo4j nodes which store Documents
        embedding_dim=384,  # default is 768
        embedding_field="embedding",
        similarity="cosine",  # "cosine" is default value for similarity
        progress_bar=False,
        create_index_if_missing=False,
        recreate_index=False,
        write_batch_size=100,
        verify_connectivity=True,  # Will try connect to Neo4j instance
    )

    print(f"Documents count: {document_store.count_documents()}")
    
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder.warm_up()

    # Step 1: Create embedding for the query
    query_embedding = text_embedder.run(query).get("embedding")
    
    if query_embedding is None:
        print("Query embedding not created successfully.")
        return
    
    print("Query embedding created successfully.")

    # Step 2: Search for similar documents using the query embedding
    similar_documents = document_store.query_by_embedding(query_embedding, top_k=3)

    if not similar_documents:
        print("No similar documents found.")
        return

    print(f"Found {len(similar_documents)} similar documents.")
    
    # Step 3: Displaying results
    for doc in similar_documents:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = doc.score
        print(f"Title: {title}\nOverview: {overview}\nScore: {score:.2f}\n{'-'*40}")

# Main function to orchestrate the entire process
def main():

    # Step 1: Create or reset vector index in Neo4j AuraDB
    create_or_reset_vector_index()

    # Step 2: Initialize Neo4j Document Store and Perform a vector search with a sample query
    query = "A hero must save the world from destruction"  # Replace with a movie plot or custom query
    perform_vector_search(query)

if __name__ == "__main__":
    main()
