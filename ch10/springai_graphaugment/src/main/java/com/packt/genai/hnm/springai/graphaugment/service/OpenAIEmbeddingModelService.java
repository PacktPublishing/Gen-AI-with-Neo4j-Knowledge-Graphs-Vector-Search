package com.packt.genai.hnm.springai.graphaugment.service;


import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OpenAIEmbeddingModelService {
    EmbeddingModel embeddingModel ;

    public OpenAIEmbeddingModelService(org.springframework.ai.embedding.EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    float[] generateEmbedding(String text) {
        float[] response = embeddingModel.embed(text) ;
        return  response ;
    }

    List<float[]> generateEmbeddingBatch(List<String> textList) {
        List<float[]> responses = embeddingModel.embed(textList) ;
        return responses ;
    }
}
