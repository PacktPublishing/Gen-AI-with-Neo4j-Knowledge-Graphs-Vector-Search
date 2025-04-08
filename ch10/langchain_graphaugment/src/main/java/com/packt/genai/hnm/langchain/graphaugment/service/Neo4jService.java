package com.packt.genai.hnm.langchain.graphaugment.service;


import com.packt.genai.hnm.langchain.graphaugment.config.Neo4jConfiguration;
import org.neo4j.driver.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
@Configuration
@EnableConfigurationProperties(Neo4jConfiguration.class)
public class Neo4jService {
    @Autowired
    private Neo4jConfiguration configuration ;

    private Driver driver ;

    public synchronized void setup() {
        if( driver == null ) {
            driver = GraphDatabase.driver(
                    configuration.getUri(),
                    AuthTokens.basic(
                            configuration.getUser(),
                            configuration.getPassword()));
            driver.verifyConnectivity();
        }
    }

    public List<EncodeRequest> getDataFromDB(String startSeason, String endSeason) {
        setup();
        String cypherTemplate = """
                MATCH (c:Customer)-[sr:%s]->(start)
                WHERE sr.embedding is null
                MATCH (c)-[:%s]->(end)
                WITH sr, start, end
                CALL {
                    WITH start, end
                    MATCH p=(start)-[:NEXT*]->(end)
                    WITH nodes(p) as txns
                    UNWIND txns as tx
                    MATCH (tx)-[:HAS_ARTICLE]->(a)
                    WITH a 
                    CALL {
                        WITH a
                        MATCH (a)-[:HAS_COLOR]->(c)
                        WITH a,c
                        MATCH (a)-[:HAS_PERCEIVED_COLOR]->(pc)
                        WITH a,c,pc
                        MATCH (a)-[:HAS_DEPARTMENT]->(d)
                        WITH a,c,pc,d
                        MATCH (a)-[:HAS_SECTION]->(s)
                        WITH a,c,pc,d,s
                        MATCH (a)-[:OF_PRODUCT]->(p)
                        RETURN ("Product: " + p.name
                           + " in  " + d.name + " Department "
                           + " in " + s.name + " Section "
                           + " with Color " + c.name
                           + " and Percived Color " + pc.name
                           + " :: Description: " + a.desc )
                           as text
                    }
                    WITH text LIMIT 10
                    WITH collect(text) as data
                    RETURN substring(reduce(out='', x in data | out + '\n' + x),1) as articles
                }
                WITH sr, articles
                RETURN elementId(sr) as elementId, articles
                LIMIT 2000
                """ ;

        String cypher = String.format(cypherTemplate, startSeason, endSeason) ;

        SessionConfig config = SessionConfig.builder().withDatabase(configuration.getDatabase()).build() ;

        try(Session session = driver.session(config)) {
            List<EncodeRequest> data = session.executeRead( tx -> {
                List<EncodeRequest> out = new ArrayList<>() ;
                var records = tx.run(cypher) ;

                while (records.hasNext()) {
                    var record = records.next() ;
                    String id = record.get("elementId").asString() ;
                    String articles = record.get("articles").asString() ;
                    out.add(new EncodeRequest(articles, id)) ;
                }

                return out ;
            }) ;
            return data ;
        }catch (Exception e) {
            e.printStackTrace();
        }
        return null ;
    }

    public List<EncodeRequest> getArticlesFromDB() {
        setup();
        String cypherTemplate = """
                MATCH (a:Article)
                WHERE a.embedding is null and a.desc is not null and trim(a.desc) <> ''
                WITH a
                CALL {
                    WITH a
                    MATCH (a)-[:HAS_COLOR]->(c)
                    WITH a,c
                    MATCH (a)-[:HAS_PERCEIVED_COLOR]->(pc)
                    WITH a,c,pc
                    MATCH (a)-[:HAS_DEPARTMENT]->(d)
                    WITH a,c,pc,d
                    MATCH (a)-[:HAS_SECTION]->(s)
                    WITH a,c,pc,d,s
                    MATCH (a)-[:OF_PRODUCT]->(p)
                    WITH a,c,pc,d,s,p
                    MATCH (p)-[:HAS_TYPE]->(t)
                    WITH a,c,pc,d,s,p,t
                    MATCH (p)-[:HAS_GROUP]->(pg)
                    RETURN ("Product : " + p.name
                        + " :: Product Group: " + pg.name
                        + " :: Product Type: " + t.name
                        + " :: Department: " + d.name
                        + " :: Section: " + s.name
                        + " :: Color: " + c.name
                        + " :: Percived Color: " + pc.name
                        + " :: Description : " + a.desc )
                        as text
                }
                RETURN elementId(a) as elementId, text as article
                """ ;

        SessionConfig config = SessionConfig.builder().withDatabase(configuration.getDatabase()).build() ;

        try(Session session = driver.session(config)) {
            List<EncodeRequest> data = session.executeRead( tx -> {
                List<EncodeRequest> out = new ArrayList<>() ;
                var records = tx.run(cypherTemplate) ;

                while (records.hasNext()) {
                    var record = records.next() ;
                    String id = record.get("elementId").asString() ;
                    String article = record.get("article").asString() ;
                    out.add(new EncodeRequest(article, id)) ;
                }

                return out ;
            }) ;
            return data ;
        }catch (Exception e) {
            e.printStackTrace();
        }
        return null ;
    }


    public void saveEmbeddings(List<Map<String, Object>> embeddings) {
        setup();
        String cypher = """
            UNWIND $data as row
            WITH row
            MATCH ()-[r]->() WHERE elementId(r) = row.id
            SET r.summary = row.summary
            WITH row, r
            CALL db.create.setRelationshipVectorProperty(r, 'embedding', row.embedding)
        """ ;
        SessionConfig config = SessionConfig.builder().withDatabase(configuration.getDatabase()).build() ;

        try(Session session = driver.session(config)) {
            session.executeWriteWithoutResult(
                    tx -> {
                        tx.run(cypher, Map.of("data", embeddings) ) ;
                    }
            ); ;
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void saveArticleEmbeddings(List<Map<String, Object>> embeddings) {
        setup();
        String cypher = """
            UNWIND $data as row
            WITH row
            MATCH (a:Article) WHERE elementId(a) = row.id
            CALL db.create.setNodeVectorProperty(a, 'embedding', row.embedding)
        """ ;
        SessionConfig config = SessionConfig.builder().withDatabase(configuration.getDatabase()).build() ;

        try(Session session = driver.session(config)) {
            session.executeWriteWithoutResult(
                    tx -> {
                        tx.run(cypher, Map.of("data", embeddings) ) ;
                    }
            ); ;
        }catch (Exception e) {
            e.printStackTrace();
        }
    }
}
