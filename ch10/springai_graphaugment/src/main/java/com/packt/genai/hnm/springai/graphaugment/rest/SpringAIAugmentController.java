package com.packt.genai.hnm.springai.graphaugment.rest;

import com.packt.genai.hnm.springai.graphaugment.config.RunConfiguration;
import com.packt.genai.hnm.springai.graphaugment.service.Neo4jService;
import com.packt.genai.hnm.springai.graphaugment.service.OpenAIChatService;
import com.packt.genai.hnm.springai.graphaugment.service.OpenAIEmbeddingModelService;
import com.packt.genai.hnm.springai.graphaugment.service.ProcessRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.UUID;

@Configuration
@EnableConfigurationProperties(RunConfiguration.class)
@RestController
public class SpringAIAugmentController {
    @Autowired
    private RunConfiguration configuration ;

    @Autowired
    private Neo4jService neo4jService ;

    @Autowired
    private OpenAIChatService chatService ;

    @Autowired
    private OpenAIEmbeddingModelService embeddingModelService ;

    private HashMap<String, ProcessRequest> currentRequests = new HashMap<>() ;

    @GetMapping("/augment/{startSeason}/{endSeason}")
    public String processAugment(
            @PathVariable(value="startSeason") String startSeason,
            @PathVariable (value="endSeason") String endSeason
    ) {
        String uuid = UUID.randomUUID().toString() ;
        ProcessRequest request = new ProcessRequest(
                chatService,
                embeddingModelService,
                neo4jService,
                configuration,
                startSeason,
                endSeason
        ) ;
        currentRequests.put(uuid, request) ;
        Thread t = new Thread(request) ;
        t.start();
        return uuid ;
    }

    @GetMapping("/augment/status/{requestId}")
    public String getStatus(
            @PathVariable (value="requestId") String requestId) {
        ProcessRequest request = currentRequests.get(requestId) ;

        if( request != null ) {
            if( request.isComplete() ) {
                currentRequests.remove(requestId) ;
            }
            return request.getCurStatus() ;
        } else {
            return "Request Not Found." ;
        }
    }
}
