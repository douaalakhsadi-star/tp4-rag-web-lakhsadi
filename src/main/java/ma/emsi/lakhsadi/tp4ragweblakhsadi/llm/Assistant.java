package ma.emsi.lakhsadi.tp4ragweblakhsadi.llm;

import dev.langchain4j.service.TokenStream;

public interface Assistant {
    TokenStream chat(String message);
}