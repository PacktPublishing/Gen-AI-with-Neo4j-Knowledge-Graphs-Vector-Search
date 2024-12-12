package com.packt.genai.hnm.langchain.graphaugment.service;

import com.packt.genai.hnm.langchain.graphaugment.config.RunConfiguration;
import dev.langchain4j.data.embedding.Embedding;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProcessRequest implements Runnable, IRequest {
    private OpenAIChatService chatService ;
    private OpenAIEmbeddingModelService embeddingModelService ;
    private Neo4jService neo4jService ;
    private RunConfiguration configuration ;

    private String startSeson ;
    private String endSeason ;

    private String curStatus = "0 %" ;

    private boolean isComplete = false ;

    public ProcessRequest(
            OpenAIChatService chatService,
            OpenAIEmbeddingModelService embeddingModelService,
            Neo4jService neo4jService,
            RunConfiguration configuration,
            String startSeson,
            String endSeason) {
        this.chatService = chatService;
        this.embeddingModelService = embeddingModelService;
        this.neo4jService = neo4jService;
        this.configuration = configuration ;
        this.startSeson = startSeson ;
        this.endSeason = endSeason ;
    }

    public String getCurStatus() {
        return curStatus ;
    }

    public boolean isComplete() {
        return isComplete;
    }

    @Override
    public void run() {
        try {
            System.out.println("Retrieving Data from Graph");
            List<EncodeRequest> dbData = neo4jService.getDataFromDB(startSeson, endSeason) ;
            System.out.println("Retrieved Data from Graph");
            int i = 0 ;
            int processingSize = dbData.size() ;
            List<Map<String, Object>> embeddings = new ArrayList<>() ;

            for( EncodeRequest request: dbData ) {
                if (i > 0 && i % configuration.getBatchSize() == 0) {
                    System.out.println("Saving Embeddings to Graph : " + i);
                    neo4jService.saveEmbeddings(embeddings);
                    embeddings.clear();
                    curStatus = ( ( i * 100.0 ) / processingSize ) + " %" ;
                }
                i++;

                Map<String, Object> embedMap = new HashMap<>();
                String id = request.getId();
                System.out.println("Retrieving Summary");
                String summary = chatService.getSummaryText(request.getText());
                System.out.println("Retrieving embedding");
                Embedding embedding = embeddingModelService.generateEmbedding(summary);
                embedMap.put("id", id);
                embedMap.put("embedding", embedding.vector());
                embedMap.put("summary", summary);
                embeddings.add(embedMap);
            }

            if( embeddings.size() > 0 ) {
                System.out.println("Saving Embeddings to Graph");
                neo4jService.saveEmbeddings(embeddings);
                embeddings.clear();
            }
            curStatus = "100 %" ;
        }catch (Exception e) {
            e.printStackTrace();
        }
        isComplete = true ;
    }
}
