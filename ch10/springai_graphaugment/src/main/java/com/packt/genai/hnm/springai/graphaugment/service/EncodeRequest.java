package com.packt.genai.hnm.springai.graphaugment.service;

public class EncodeRequest {
    private long id ;
    private String text ;

    public EncodeRequest(String text, long id) {
        this.text = text;
        this.id = id;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}
