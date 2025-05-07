package com.packt.genai.hnm.langchain.graphaugment.service;

import org.springframework.stereotype.Service;

@Service
public class OpenAIChatService {

    private ChatAssistant assistant ;

    public OpenAIChatService(ChatAssistant assistant) {
        this.assistant = assistant;
    }

    public String getSummaryText(String input) {
        String out = assistant.chat(input) ;
//        System.out.println("Chat out : " + out);
        return out ;
    }
}
