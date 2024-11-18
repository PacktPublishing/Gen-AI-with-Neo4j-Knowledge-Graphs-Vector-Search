from neo4j import GraphDatabase

# Define the connection credentials for the Neo4j database
uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
username = "neo4j"             # Replace with your Neo4j username
password = "password"          # Replace with your Neo4j password

# Create a driver instance to connect to the Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))

def project_graph():
    """
    Project the graph in memory using GDS with weighted relationships.
    """
    query = """
    CALL gds.graph.project(
      'movieGraph',
      'Movie',
      {
        HAS_PLOT: {
          type: 'HAS_PLOT',
          orientation: 'NATURAL',
          properties: 'weight'
        }
      }
    );
    """
    with driver.session() as session:
        session.run(query)
        print("Graph projected in memory.")

def run_pagerank():
    """
    Run the PageRank algorithm on the projected graph.
    """
    query = """
    CALL gds.pageRank.stream('movieGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).title AS movie, score
    ORDER BY score DESC;
    """
    with driver.session() as session:
        result = session.run(query)
        print("PageRank Results:")
        for record in result:
            print(f"Movie: {record['movie']}, PageRank: {record['score']}")

def drop_graph():
    """
    Drop the in-memory graph projection to free up resources.
    """
    query = """
    CALL gds.graph.drop('movieGraph') YIELD graphName;
    """
    with driver.session() as session:
        session.run(query)
        print("Graph projection dropped.")

# Add relationship weights (optional)
def add_relationship_weights():
    """
    Add weights to relationships in the Neo4j database for better PageRank results.
    """
    query = """
    MATCH (m1:Movie)-[r:HAS_PLOT]->(p:Plot)
    SET r.weight = 1.0;
    """
    with driver.session() as session:
        session.run(query)
        print("Weights added to relationships.")

# Execute the workflow
add_relationship_weights()  # Step 1: Add weights (if needed)
project_graph()  # Step 2: Project the graph
run_pagerank()   # Step 3: Run PageRank
drop_graph()     # Step 4: Drop the graph

# Close the database driver
driver.close()
