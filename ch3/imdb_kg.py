from neo4j import GraphDatabase

# Define the connection credentials for the Neo4j database
uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
username = "neo4j"             # Replace with your Neo4j username
password = "password"          # Replace with your Neo4j password

# Create a driver instance to connect to the Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))

# Define a function to create the knowledge graph
def create_graph(tx):
    # Create movie nodes
    tx.run("CREATE (m:Movie {title: 'The Matrix', year: 1999})")
    tx.run("CREATE (m:Movie {title: 'Inception', year: 2010})")
    tx.run("CREATE (m:Movie {title: 'Interstellar', year: 2014})")
    tx.run("CREATE (m:Movie {title: 'The Dark Knight', year: 2008})")
    tx.run("CREATE (m:Movie {title: 'Pulp Fiction', year: 1994})")
    
    # Create plot nodes
    tx.run("CREATE (p:Plot {description: 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.'})")
    tx.run("CREATE (p:Plot {description: 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.'})")
    tx.run("CREATE (p:Plot {description: 'A team of explorers travels through a wormhole in space in an attempt to ensure humanity’s survival.'})")
    tx.run("CREATE (p:Plot {description: 'When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.'})")
    tx.run("CREATE (p:Plot {description: 'The lives of two mob hitmen, a boxer, a gangster, and his wife intertwine in four tales of violence and redemption.'})")
    
    # Create relationships between movies and their plots
    tx.run("""
    MATCH (m:Movie {title: 'The Matrix'}), 
          (p:Plot {description: 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.'})
    CREATE (m)-[:HAS_PLOT]->(p)
    """)
    tx.run("""
    MATCH (m:Movie {title: 'Inception'}), 
          (p:Plot {description: 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.'})
    CREATE (m)-[:HAS_PLOT]->(p)
    """)
    tx.run("""
    MATCH (m:Movie {title: 'Interstellar'}), 
          (p:Plot {description: 'A team of explorers travels through a wormhole in space in an attempt to ensure humanity’s survival.'})
    CREATE (m)-[:HAS_PLOT]->(p)
    """)
    tx.run("""
    MATCH (m:Movie {title: 'The Dark Knight'}), 
          (p:Plot {description: 'When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.'})
    CREATE (m)-[:HAS_PLOT]->(p)
    """)
    tx.run("""
    MATCH (m:Movie {title: 'Pulp Fiction'}), 
          (p:Plot {description: 'The lives of two mob hitmen, a boxer, a gangster, and his wife intertwine in four tales of violence and redemption.'})
    CREATE (m)-[:HAS_PLOT]->(p)
    """)

# Define a function to query the knowledge graph
def query_graph(tx):
    # Query to retrieve movies and their plots
    result = tx.run("""
    MATCH (m:Movie)-[:HAS_PLOT]->(p:Plot)
    RETURN m.title AS movie, m.year AS year, p.description AS plot
    """)
    # Print the results
    for record in result:
        print(f"Movie: {record['movie']} ({record['year']}) - Plot: {record['plot']}")

# Establish a session and write to the database
with driver.session() as session:
    session.execute_write(create_graph)

# Establish a session and read from the database
with driver.session() as session:
    session.execute_read(query_graph)

# Close the driver connection
driver.close()