import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

class CreateGraph:

    def __init__(self, uri, user, password, database='neo4j'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

    def close(self):
        self.driver.close()

    def db_cleanup(self):
        print("Doing Database Cleanup.")
        query = """
        MATCH (n) DETACH DELETE (n)
        """
        with self.driver.session() as session:
            session.run(query)
            print("Database Cleanup Done. Using blank database.")

    def create_constraints_indexes(self):
        queries = [
            "CREATE CONSTRAINT unique_tmdb_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdbId IS UNIQUE;",
            "CREATE CONSTRAINT unique_movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE;",
            "CREATE CONSTRAINT unique_prod_id IF NOT EXISTS FOR (p:ProductionCompany) REQUIRE p.company_id IS UNIQUE;",
            "CREATE CONSTRAINT unique_genre_id IF NOT EXISTS FOR (g:Genre) REQUIRE g.genre_id IS UNIQUE;",
            "CREATE CONSTRAINT unique_lang_id IF NOT EXISTS FOR (l:SpokenLanguage) REQUIRE l.language_code IS UNIQUE;",
            "CREATE CONSTRAINT unique_country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.country_code IS UNIQUE;",
            "CREATE INDEX actor_id IF NOT EXISTS FOR (p:Person) ON (p.actor_id);",
            "CREATE INDEX crew_id IF NOT EXISTS FOR (p:Person) ON (p.crew_id);",
            "CREATE INDEX movieId IF NOT EXISTS FOR (m:Movie) ON (m.movieId);",
            "CREATE INDEX user_id IF NOT EXISTS FOR (p:Person) ON (p.user_id);"
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
            print("Constraints and Indexes created successfully.")


    def load_movies(self, csv_file, limit):
        query = f"""
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        WITH row, toInteger(row.tmdbId) AS tmdbId
        WHERE tmdbId IS NOT NULL
        WITH row, tmdbId
        LIMIT {limit}
        MERGE (m:Movie {{tmdbId: tmdbId}})
        ON CREATE SET m.title = coalesce(row.title, "None"),
                      m.original_title = coalesce(row.original_title, "None"),
                      m.adult = CASE 
                                    WHEN toInteger(row.adult) = 1 THEN 'Yes' 
                                    ELSE 'No' 
                                END,
                      m.budget = toInteger(coalesce(row.budget, 0)),
                      m.original_language = coalesce(row.original_language, "None"),
                      m.revenue = toInteger(coalesce(row.revenue, 0)),
                      m.tagline = coalesce(row.tagline, "None"),
                      m.overview = coalesce(row.overview, "None"),
                      m.release_date = coalesce(row.release_date, "None"),
                      m.runtime = toFloat(coalesce(row.runtime, 0)),
                      m.belongs_to_collection = coalesce(row.belongs_to_collection, "None");
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Movies loaded from {csv_file} (limited to {limit} entries)")

    def load_genres(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        WITH m, row
        MERGE (g:Genre {genre_id: toInteger(row.genre_id)})
        ON CREATE SET g.genre_name = row.genre_name
        MERGE (m)-[:HAS_GENRE]->(g);
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Genres and relationships to movies loaded from {csv_file}")

    def load_production_companies(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        WITH m, row
        MERGE (pc:ProductionCompany {company_id: toInteger(row.company_id)})
        ON CREATE SET pc.company_name = row.company_name
        MERGE (m)-[:PRODUCED_BY]->(pc);
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Production companies and relationships to movies loaded from {csv_file}")

    def load_production_countries(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        WITH m, row
        MERGE (c:Country {country_code: row.country_code})
        ON CREATE SET c.country_name = row.country_name
        MERGE (m)-[:PRODUCED_IN]->(c);
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Production countries and relationships to movies loaded from {csv_file}")

    def load_spoken_languages(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        WITH m, row
        MERGE (l:SpokenLanguage {language_code: row.language_code})
        ON CREATE SET l.language_name = row.language_name
        MERGE (m)-[:HAS_LANGUAGE]->(l);
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Spoken languages and relationships to movies loaded from {csv_file}")

    def load_keywords(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        SET m.keywords = row.keywords;
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Keywords loaded from {csv_file}")

    def load_person_actors(self, csv_file):
        query1 = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        CALL (row){
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        WITH m, row
        MERGE (p:Person {actor_id: toInteger(row.actor_id)})
        ON CREATE SET p.name = row.name, p.role= 'actor'
        MERGE (p)-[a:ACTED_IN]->(m)
        ON CREATE SET a.character = coalesce(row.character, "None"), a.cast_id= toInteger(row.cast_id)
        }IN TRANSACTIONS OF 50000 ROWS;
        """
        with self.driver.session() as session:
            session.run(query1, csvFile=f'{csv_file}')
            print(f"Actors loaded from {csv_file}")
        query2 = """
        MATCH (n:Person) WHERE n.role="actor" SET n:Actor
        """
        with self.driver.session() as session:
            session.run(query2)
            print(f"Actor label created additionally")

    def load_person_crew(self, csv_file):
        query1 = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        MERGE (p:Person {crew_id: toInteger(row.crew_id)})
        ON CREATE SET p.name = row.name, p.role = row.job
        WITH p, m, row,
        CASE
        WHEN row.job='Director' THEN "DIRECTED"
        WHEN row.job='Producer' THEN "PRODUCED"
        ELSE "Unknown"
        END AS crew_rel
        CALL apoc.create.relationship(p, crew_rel, {}, m)
        YIELD rel
        RETURN rel;
        """
        with self.driver.session() as session:
            session.run(query1, csvFile=f'{csv_file}')
            print(f"Directors and Producers loaded from {csv_file}")
        query2 = """
        MATCH (n:Person) WHERE n.role="Director" SET n:Director
        """
        with self.driver.session() as session:
            session.run(query2)
            print(f"Director label created additionally")
        query3 = """
        MATCH (n:Person) WHERE n.role="Producer" SET n:Producer
        """
        with self.driver.session() as session:
            session.run(query3)
            print(f"Producer label created additionally")


    def load_links(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})  // Check if the movie exists
        SET m.movieId = toInteger(row.movieId),
            m.imdbId = row.imdbId;
        """
        with self.driver.session() as session:
            session.run(query, csvFile=f'{csv_file}')
            print(f"Links loaded from {csv_file}")


    def load_ratings(self, csv_file):
        query1 = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        CALL (row){
        MATCH (m:Movie {movieId: toInteger(row.movieId)})  // Check if the movie exists
        WITH m, row
        MERGE (p:Person {user_id: toInteger(row.userId)})
        ON CREATE SET p.role= 'user'
        MERGE (p)-[r:RATED]->(m)
        ON CREATE SET r.rating = toFloat(row.rating), r.timestamp = toInteger(row.timestamp)
        }IN TRANSACTIONS OF 50000 ROWS;
        """
        with self.driver.session() as session:
            session.run(query1, csvFile=f'{csv_file}')
            print(f"Ratings loaded from {csv_file}")
        query2 = """
        MATCH (n:Person) WHERE n.role="user" SET n:User
        """
        with self.driver.session() as session:
            session.run(query2)
            print(f"User label created additionally")



def main():
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    graph = CreateGraph(uri, user, password)

    graph.db_cleanup()
    graph.create_constraints_indexes()

    # Load data from CSV files with a limit on entries for movies
    movie_limit = 10000  # Limit only applied to movies
    graph.load_movies('https://storage.googleapis.com/movies-packt/normalized_movies.csv', movie_limit)

    # Load related nodes and create relationships conditionally
    graph.load_genres('https://storage.googleapis.com/movies-packt/normalized_genres.csv')
    graph.load_production_companies('https://storage.googleapis.com/movies-packt/normalized_production_companies.csv')
    graph.load_production_countries('https://storage.googleapis.com/movies-packt/normalized_production_countries.csv')
    graph.load_spoken_languages('https://storage.googleapis.com/movies-packt/normalized_spoken_languages.csv')
    graph.load_keywords('https://storage.googleapis.com/movies-packt/normalized_keywords.csv')
    graph.load_person_actors('https://storage.googleapis.com/movies-packt/normalized_cast.csv')
    graph.load_person_crew('https://storage.googleapis.com/movies-packt/normalized_crew.csv')
    graph.load_links('https://storage.googleapis.com/movies-packt/normalized_links.csv')
    graph.load_ratings('https://storage.googleapis.com/movies-packt/normalized_ratings_small.csv')


    graph.close()

if __name__ == "__main__":
    main()
