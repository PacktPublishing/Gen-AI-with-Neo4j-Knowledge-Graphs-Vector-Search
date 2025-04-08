import os
import openai
import gradio as gr
from dotenv import load_dotenv
from neo4j import GraphDatabase
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret
from haystack import Pipeline
from neo4j_haystack import (
    Neo4jDynamicDocumentRetriever,
    Neo4jClientConfig,
)

# Load environment variables
load_dotenv()

# Neo4j details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Vector index setup
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
def create_or_reset_vector_index():
    with driver.session() as session:
        session.run("DROP INDEX overview_embeddings IF EXISTS")
        session.run("""
            CREATE VECTOR INDEX overview_embeddings IF NOT EXISTS
            FOR (m:Movie) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,  
                `vector.similarity_function`: 'cosine'}}
        """)
        print("Vector index created or reset.")

# Client config
client_config = Neo4jClientConfig(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
)

# Conversational chatbot handler using Cypher-powered search
def perform_vector_search_cypher(user_input):
    print("🔍 MESSAGES RECEIVED:", user_input)

    cypher_query = """
        CALL db.index.vector.queryNodes("overview_embeddings", $top_k, $query_embedding)
        YIELD node AS movie, score
        MATCH (movie:Movie)
        RETURN movie.title AS title, movie.overview AS overview, score
    """

    # Embedder
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )

    # Retriever
    retriever = Neo4jDynamicDocumentRetriever(
        client_config=client_config,
        runtime_parameters=["query_embedding"],
        compose_doc_from_result=True,
        verify_connectivity=True,
    )

    # Pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    # Run pipeline
    result = pipeline.run({
        "query_embedder": {"text": user_input},
        "retriever": {
            "query": cypher_query,
            "parameters": {"index": "overview_embeddings", "top_k": 3},
        },
    })

    documents = result["retriever"]["documents"]

    if not documents:
        return "I couldn’t find anything relevant."

    reply = ""
    for doc in documents:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = getattr(doc, "score", None)
        reply += f"🎬 **{title}**\n{overview}\n(score: {score:.2f})\n\n"

    return reply.strip()

def conversational_chatbot(user_input):
    return perform_vector_search_cypher(user_input)

# Gradio Chat Interface setup
chat_interface = gr.Interface(
    fn=conversational_chatbot, 
    inputs=gr.Textbox(
        placeholder="What kind of movie would you like to watch?",
        lines=3,
        label="Your movie preference"
    ),
    outputs=gr.Textbox(
        label="Recommendations",
        lines=12
    ),
    title="AI Movie Recommendation System",
    description="Ask me about movies! I can recommend movies based on your preferences.",
    examples=[
        ["I want to watch a sci-fi movie with time travel"],
        ["Recommend me a romantic comedy with a happy ending"],
        ["I'm in the mood for something with superheroes but not too serious"],
        ["I want a thriller that keeps me on the edge of my seat"],
        ["Show me movies about artificial intelligence taking over the world"]
    ],
    flagging_mode="never"
)
# Main
def main():
    create_or_reset_vector_index()
    chat_interface.launch()

if __name__ == "__main__":
    main()
