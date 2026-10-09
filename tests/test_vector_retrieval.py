from gtm_agent.models.schemas import ProductBrief
from gtm_agent.retrieval.vector import VectorRetriever, get_embedder


def test_vector_store_persists_collection_and_source_metadata(tmp_path) -> None:
    documents = [
        ProductBrief(
            title="Product",
            source_id="product.md",
            content="LaunchPad coordinates product launch campaigns for marketing teams.",
        ),
        ProductBrief(
            title="Calendar",
            source_id="calendar.csv",
            content="LaunchPad launch date is November 1, 2026. Pricing is INR 4,999.",
            media_type="spreadsheet",
        ),
    ]
    first = VectorRetriever(documents, persist_directory=tmp_path / "chroma", backend="hash")
    results = first.search("LaunchPad launch date pricing marketing", min_score=0.05)
    second = VectorRetriever(documents, persist_directory=tmp_path / "chroma", backend="hash")
    assert first.collection_name == second.collection_name
    assert second._collection.count() == 2
    assert {item.source_id for item in results} == {"product.md", "calendar.csv"}


def test_minilm_semantic_retrieval_matches_paraphrase(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GTM_EMBEDDINGS_LOCAL_ONLY", "true")
    get_embedder.cache_clear()
    documents = [
        ProductBrief(
            title="LaunchPad",
            source_id="product.md",
            content="LaunchPad lets go-to-market leaders coordinate release announcements across channels.",
        ),
        ProductBrief(
            title="Unrelated",
            source_id="facilities.md",
            content="The office cafeteria serves lunch from noon until two o'clock.",
        ),
    ]
    retriever = VectorRetriever(
        documents,
        persist_directory=tmp_path / "semantic",
        backend="sentence-transformer",
    )
    results = retriever.search("How can marketing teams plan a product launch?", limit=1, min_score=0.1)
    assert results
    assert results[0].source_id == "product.md"
    assert retriever.mode == "vector"
    get_embedder.cache_clear()
