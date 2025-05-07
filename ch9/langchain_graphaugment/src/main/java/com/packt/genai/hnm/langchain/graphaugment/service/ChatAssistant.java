package com.packt.genai.hnm.langchain.graphaugment.service;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import dev.langchain4j.service.spring.AiService;

@AiService
public interface ChatAssistant {

    @SystemMessage("""
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
            
            Data:
            {text}
    """)
    String chat(String text);
}
