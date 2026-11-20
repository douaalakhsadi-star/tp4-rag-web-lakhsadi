package ma.emsi.lakhsadi.tp4ragweblakhsadi.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;

@ApplicationScoped
public class SourcesRag {

    private ContentRetriever recuperateurRag;
    private ContentRetriever recuperateurIot;

    @PostConstruct
    public void initialiser() {
        EmbeddingModel modeleEmbedding = new BgeSmallEnV15EmbeddingModel();

        recuperateurRag = creerRecuperateur("rag.pdf", modeleEmbedding);
        recuperateurIot = creerRecuperateur(
                "atelier_6_iot_email.pdf",
                modeleEmbedding
        );
    }

    private ContentRetriever creerRecuperateur(
            String fichier,
            EmbeddingModel modeleEmbedding) {

        Document document = ClassPathDocumentLoader.loadDocument(
                fichier,
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments = DocumentSplitters
                .recursive(1_000, 100)
                .split(document);

        List<Embedding> embeddings = modeleEmbedding
                .embedAll(segments)
                .content();

        EmbeddingStore<TextSegment> magasin = new InMemoryEmbeddingStore<>();
        magasin.addAll(embeddings, segments);

        System.out.println(
                fichier + " : " + segments.size() + " passages enregistrés."
        );

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(magasin)
                .embeddingModel(modeleEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public ContentRetriever getRecuperateurRag() {
        return recuperateurRag;
    }

    public ContentRetriever getRecuperateurIot() {
        return recuperateurIot;
    }
}