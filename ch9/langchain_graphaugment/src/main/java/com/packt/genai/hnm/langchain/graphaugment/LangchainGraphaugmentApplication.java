package com.packt.genai.hnm.langchain.graphaugment;

import com.packt.genai.hnm.langchain.graphaugment.config.Neo4jConfiguration;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
public class LangchainGraphaugmentApplication {

	public static void main(String[] args) {
		SpringApplication.run(LangchainGraphaugmentApplication.class, args);
	}

}
