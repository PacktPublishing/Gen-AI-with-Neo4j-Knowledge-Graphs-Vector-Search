import os
import openai
from neo4j_haystack import Neo4jDocumentStore, Neo4jClientConfig, Neo4jEmbeddingRetriever
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret
from haystack import Pipeline, Document
# from haystack.schema import Filter
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
openai.api_key = OPENAI_API_KEY

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Neo4j Document Store and Haystack Components
client_config = Neo4jClientConfig(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

document_store = Neo4jDocumentStore(
    client_config=client_config,
    index="overview_embeddings",  # The name of the Vector Index in Neo4j
    node_label="Movie",  # Label to Neo4j nodes which store Documents
    embedding_dim=1536,  # Dimension of embeddings (for OpenAI ADA it's 1536)
    embedding_field="embedding",
    similarity="cosine",  # Cosine similarity for vector search
    verify_connectivity=True,
)


# Initialize Haystack's OpenAITextEmbedder for creating embeddings
text_embedder = OpenAITextEmbedder(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)


def fetch_multi_hop_related_movies(title):
    query = """
    MATCH (m:Movie {title: $title})<-[:ACTED_IN|DIRECTED]-(p)-[:ACTED_IN|DIRECTED]->(related:Movie)
    WITH related.title AS related_movie, p.name AS person, 
         CASE 
            WHEN (p)-[:ACTED_IN]->(m) AND (p)-[:ACTED_IN]->(related) THEN 'Actor'
            WHEN (p)-[:DIRECTED]->(m) AND (p)-[:DIRECTED]->(related) THEN 'Director'
            ELSE 'Unknown Role'
         END AS role, 
         related.overview AS overview, related.embedding AS embedding
     RETURN related_movie, person, role, overview, embedding
    """
    with driver.session() as session:
        result = session.run(query, title=title)
        documents = []
        for record in result:
            documents.append(
                Document(
                    content=record.get("overview", "No overview available"),  # Store overview in content
                    meta={
                        "title": record.get("related_movie", "Unknown Movie"),  # Movie title
                        "person": record.get("person", "Unknown Person"),       # Actor/Director's name
                        "role": record.get("role", "Unknown Role"),              # Actor or Director
                        "embedding": record.get("embedding", "No embedding available")  # Retrieve the precomputed embedding
                    },
                )
            )
    return documents

def fetch_related_movies_via_actors_and_directors(query, movie_title):
    # Fetch multi-hop related movies from Neo4j
    multi_hop_docs = fetch_multi_hop_related_movies(movie_title)

    if not multi_hop_docs:
        print(f"No related movies found for {movie_title}")
        return

    document_store.write_documents(multi_hop_docs)

    # Generate embedding for the search query (e.g., "Find movies with shared actors and directors")
    query_embedding = text_embedder.run(query).get("embedding")


    if query_embedding is None:
        print("Query embedding not created successfully.")
        return

    # Perform vector search only on the multi-hop related movies
    similar_docs = document_store.query_by_embedding(query_embedding, top_k=3)

    if not similar_docs:
        print("No similar documents found.")
        return

    # Display the top-matching results
    for doc in similar_docs:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = doc.score
        print(f"Title: {title}\nOverview: {overview}\nScore: {score:.2f}\n{'-'*40}")
    print("\n\n")


# Main function to execute all use cases
def main():
    movie_title = "Jurassic Park"
    search_query = "Find movies with shared actors and directors about dinosaurs"

    print("=== Traversing Multiple Relationships to Reveal Hidden Insights ===")
    fetch_related_movies_via_actors_and_directors(search_query, movie_title)

if __name__ == "__main__":
    main()