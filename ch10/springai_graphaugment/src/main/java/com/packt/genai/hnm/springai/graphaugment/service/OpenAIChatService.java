package com.packt.genai.hnm.springai.graphaugment.service;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class OpenAIChatService {
    private final ChatClient chatClient;

    private final String SYSTEM_PROMPT_TEMPLATE = """
             ---Role---
             
             You are an helpful assistant with expertise in fashion for a clothing company.
             
             ---Goal---
             
            Your goal is to generate a summary of the products purchased by the customers and descriptions of each fo the products.\s
            Your summary should contain two sections -\s
            Section 1 - Overall summary outlining the fashion preferences of the customer based on the purchases. Limit the summary to 3 sentences
            Section 2 - highlight 3-5 individual purchases.
            
            You should use the data provided in the section below as the primary context for generating the response.\s
            If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so.\s
            Do not make anything up.
            
             Data Description:
            - Each Customer has an ID. Customer ID is a numeric value.
            - Each Customer has purchased more than one clothing articles (products). Products have descriptions.
            - The order of the purchases is very important. You should take into account the order when generating the summary.
        
            Response:
            ---
            # Overall Fashion Summary:
        
            \\n\\n
        
            # Individual Purchase Details:
    --
    """ ;

    private final String userMessage = """

            Data:
            {data}
            """ ;

    public OpenAIChatService(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    public String getSummaryText(String input) {
//        String out = assistant.chat(input) ;
//        return out ;
//        var systemMessage = new SystemPromptTemplate(SYSTEM_PROMPT_TEMPLATE)
//                .createMessage(Map.of("data", input));
        ChatResponse response = chatClient
                .prompt()
                .system(SYSTEM_PROMPT_TEMPLATE)
                .user(p -> p.text(userMessage).param("data", input))
                .call()
                .chatResponse() ;

        return response.getResult().getOutput().getContent() ;
    }

}
