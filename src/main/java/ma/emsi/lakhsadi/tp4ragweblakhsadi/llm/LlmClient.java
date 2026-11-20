package ma.emsi.lakhsadi.tp4ragweblakhsadi.llm;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiStreamingChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

@Dependent
public class LlmClient implements Serializable {

    private static final Logger LOGGER =
            Logger.getLogger(LlmClient.class.getName());

    @Inject
    private CompteurTokens compteurTokens;

    @Inject
    private SourcesRag sourcesRag;

    private String systemRole;
    private ChatMemory chatMemory;
    private Assistant assistant;

    @PostConstruct
    public void initialiser() {
        LOGGER.warning("DIAGNOSTIC - Initialisation LlmClient version 2");

        String cle = System.getenv("GEMINI_KEY");

        ChatModel modeleRoutage = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        StreamingChatModel modeleReponse =
                GoogleGenAiStreamingChatModel.builder()
                        .apiKey(cle)
                        .modelName("gemini-3.5-flash-lite")
                        .build();

        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(
                sourcesRag.getRecuperateurRag(),
                "Support de cours sur le fine-tuning et le RAG : "
                        + "modèles de langage, embeddings, bases vectorielles, "
                        + "recherche sémantique et LangChain4j."
        );

        sources.put(
                sourcesRag.getRecuperateurIot(),
                "Atelier sur un pipeline IoT en temps réel : "
                        + "ESP32, DHT22, Wokwi, MQTT, HiveMQ, Node-RED, "
                        + "InfluxDB Cloud, Grafana Cloud et alertes par email."
        );

        QueryRouter routeurModele =
                new LanguageModelQueryRouter(modeleRoutage, sources);

        QueryRouter routeur = query -> {
            // Niveau WARNING temporaire pour vérifier le routage.
            LOGGER.warning(
                    "DIAGNOSTIC - Question pour le routage : " + query.text()
            );

            var selection = routeurModele.route(query);

            if (selection.contains(sourcesRag.getRecuperateurRag())) {
                LOGGER.warning(
                        "DIAGNOSTIC - Source sélectionnée : PDF RAG"
                );
            }

            if (selection.contains(sourcesRag.getRecuperateurIot())) {
                LOGGER.warning(
                        "DIAGNOSTIC - Source sélectionnée : PDF IoT"
                );
            }

            if (selection.isEmpty()) {
                LOGGER.warning(
                        "DIAGNOSTIC - Aucune source sélectionnée"
                );
            }

            return selection;
        };

        RetrievalAugmentor augmentateur = DefaultRetrievalAugmentor.builder()
                .queryRouter(routeur)
                .build();

        chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        assistant = AiServices.builder(Assistant.class)
                .streamingChatModel(modeleReponse)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentateur)
                .build();
    }

    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;
        chatMemory.clear();

        if (systemRole != null && !systemRole.isBlank()) {
            chatMemory.add(SystemMessage.from(systemRole));
        }
    }

    public String getSystemRole() {
        return systemRole;
    }

    public TokenStream envoyerQuestion(String question) {
        return assistant.chat(question);
    }

    public void compterTokens(ChatResponse reponse) {
        if (reponse.tokenUsage() != null
                && reponse.tokenUsage().totalTokenCount() != null) {
            compteurTokens.ajouter(
                    reponse.tokenUsage().totalTokenCount()
            );
        }
    }
}