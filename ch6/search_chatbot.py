import os
import openai
from neo4j_haystack import Neo4jDocumentStore, Neo4jDynamicDocumentRetriever, Neo4jClientConfig
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret
from haystack import GeneratedAnswer, Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from neo4j import GraphDatabase
from dotenv import load_dotenv
import gradio as gr

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
            `vector.dimensions`: 1536,  
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


# Perform vector search using Haystack and without Cypher
def perform_vector_search(query):


    print("Performing vector search using Haystack and without Cypher")
    document_store = Neo4jDocumentStore(
        client_config=client_config,
        index="overview_embeddings",  # The name of the Vector Index in Neo4j
        node_label="Movie",  # Providing a label to Neo4j nodes which store Documents
        embedding_dim=1536,  # default is 768
        embedding_field="embedding",
        similarity="cosine",  # "cosine" is default value for similarity
        progress_bar=False,
        create_index_if_missing=False,
        recreate_index=False,
        write_batch_size=100,
        verify_connectivity=True,  # Will try connect to Neo4j instance
    )

    print(f"Documents count: {document_store.count_documents()}")


    # # Initialize document store (In-memory for now, but you can configure other stores)
    # document_store = InMemoryDocumentStore()

    
    text_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )

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

    print("\n\n")
    
    # Step 3: Format results for return
    result_output = ""
    for doc in similar_documents:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = doc.score
        score_display = f"{score:.2f}" if score is not None else "N/A"
        result_output += f"Title: {title}\nOverview: {overview}\nScore: {score_display}\n{'-'*40}\n"
    
    return result_output



# Perform vector search using Haystack and Cypher
def perform_vector_search_cypher(query):


    print("Performing vector search using Haystack and Cypher")
    
    # cypher_query = """
    #         CALL db.index.vector.queryNodes($index, $top_k, $query_embedding)
    #         YIELD node as similarMovie, score
    #         MATCH (similarMovie:Movie)
    #         RETURN similarMovie {.*, embedding:null} as movie, [(similarMovie)<-[rel]-(p:Person) |  {role:type(rel), name: p.name} ] as cast, 
    #          [(similarMovie)-[:HAS_GENRE]->(g:Genre) |  g.genre_name ] as genres, score
    #         LIMIT 5
    #     """

    cypher_query = """
            CALL db.index.vector.queryNodes("overview_embeddings", $top_k, $query_embedding)
            YIELD node AS movie, score
            MATCH (movie:Movie)
            RETURN movie.title AS title, movie.overview AS overview, score
        """

    text_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )


    retriever = Neo4jDynamicDocumentRetriever(
        # client_config=client_config, runtime_parameters=["query_embedding"], doc_node_name="movie", verify_connectivity=True,
        client_config=client_config, runtime_parameters=["query_embedding"], compose_doc_from_result=True, verify_connectivity=True,
    )

    pipeline = Pipeline()
    pipeline.add_component("query_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    result = pipeline.run(
        {
            "query_embedder": {"text": query},
            "retriever": {
                "query": cypher_query,
                "parameters": {"index": "overview_embeddings", "top_k": 3},
            },
        }
    )

    # # Print the raw result to debug
    # print("Raw retriever result:", result)


    # Extracting documents from the retriever results
    documents = result["retriever"]["documents"]

    # Step 3: Displaying results
    for doc in documents:
        # Extract title and overview from document metadata
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        
        # Extract score from the document (not from meta)
        score = getattr(doc, "score", None)

        # Format score if it exists, else show "N/A"
        score_display = f"{score:.2f}" if score is not None else "N/A"

        # Print the title, overview, and score (or N/A for missing score)
        print(f"Title: {title}\nOverview: {overview}\nScore: {score_display}\n{'-'*40}\n")


# Define the Gradio chatbot interface
def chatbot(query):
    return perform_vector_search(query)

# Main function to orchestrate the entire process
def main():
    # Step 1: Create or reset vector index in Neo4j AuraDB
    create_or_reset_vector_index()

    # Step 2: Launch Gradio chatbot interface
    gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Movie Search Chatbot", description="Ask me about movies!").launch()

if __name__ == "__main__":
    main()


# Define the Gradio chatbot interface
def chatbot(query):
    return perform_vector_search(query)

# Main function to orchestrate the entire process
def main():
    # Step 1: Create or reset vector index in Neo4j AuraDB
    create_or_reset_vector_index()

    # Step 2: Launch Gradio chatbot interface
    gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Movie Search Chatbot", description="Ask me about movies!").launch()

if __name__ == "__main__":
    main()
