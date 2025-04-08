package com.packt.genai.hnm.langchain.graphaugment.service;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import org.springframework.stereotype.Service;
import dev.langchain4j.model.output.Response;

import java.util.List;

@Service
public class OpenAIEmbeddingModelService {
    EmbeddingModel embeddingModel ;

    public OpenAIEmbeddingModelService(EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    Embedding generateEmbedding(String text) {
        Response<Embedding> response = embeddingModel.embed(text) ;
        return  response.content() ;
    }

    List<Embedding> generateEmbeddingBatch(List<TextSegment> textList) {
        Response<List<Embedding>> responses = embeddingModel.embedAll(textList) ;
        return responses.content() ;
    }
}
