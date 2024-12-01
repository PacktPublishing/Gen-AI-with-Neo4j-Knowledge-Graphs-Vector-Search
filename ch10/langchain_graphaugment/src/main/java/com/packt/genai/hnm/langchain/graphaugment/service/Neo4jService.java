package com.packt.genai.hnm.langchain.graphaugment.service;


import com.packt.genai.hnm.langchain.graphaugment.config.Neo4jConfiguration;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
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

//    private EmbeddingStore<TextSegment> embeddingStore ;

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

    public void test() {
        System.out.println(configuration.getUri()) ;
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
                    WITH collect(a.desc) as data
                    RETURN substring(reduce(out='', x in data | out + ', ' + x),1) as articles
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
}
