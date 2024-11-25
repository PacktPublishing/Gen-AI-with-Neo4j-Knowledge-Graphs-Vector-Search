package com.packt.genai.hnm.langchain.graphaugment.service;

public class EncodeRequest {
    private String id ;
    private String text ;

    public EncodeRequest(String text, String id) {
        this.text = text;
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}
