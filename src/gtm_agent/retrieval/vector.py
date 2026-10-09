from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import chromadb
import numpy as np
from chromadb.config import Settings

from gtm_agent.models.schemas import EvidencePassage, ProductBrief
from gtm_agent.retrieval.local import LexicalRetriever, chunk_brief, tokenize


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        local_only = os.getenv("GTM_EMBEDDINGS_LOCAL_ONLY", "true").lower() == "true"
        self.model = SentenceTransformer(self.name, local_files_only=local_only)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32).tolist()


class HashingEmbedder:
    """Fast deterministic test/fallback embedder; production defaults to MiniLM."""

    name = "deterministic-hash-256"

    def encode(self, texts: list[str]) -> list[list[float]]:
        encoded: list[list[float]] = []
        for text in texts:
            vector = np.zeros(256, dtype=np.float32)
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:2], "big") % len(vector)] += 1.0
            norm = float(np.linalg.norm(vector))
            encoded.append((vector / norm if norm else vector).tolist())
        return encoded


@lru_cache(maxsize=2)
def get_embedder(backend: str | None = None) -> Embedder:
    selected = backend or os.getenv("GTM_EMBEDDING_BACKEND", "sentence-transformer")
    return HashingEmbedder() if selected == "hash" else SentenceTransformerEmbedder()


class VectorRetriever:
    """Chroma vector index with local embeddings and an attributed lexical fallback."""

    def __init__(
        self,
        documents: list[ProductBrief],
        *,
        persist_directory: str | Path | None = None,
        backend: str | None = None,
    ) -> None:
        if not documents:
            raise ValueError("at least one source document is required")
        self.documents = documents
        self.mode = "vector"
        self.error: str | None = None
        self._lexical = [LexicalRetriever(document) for document in documents]
        self._embedder = get_embedder(backend)
        settings = Settings(anonymized_telemetry=False)
        if persist_directory:
            root = Path(persist_directory)
            root.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(root), settings=settings)
        else:
            self._client = chromadb.EphemeralClient(settings=settings)

        fingerprint = hashlib.sha256()
        fingerprint.update(self._embedder.name.encode())
        for document in documents:
            fingerprint.update(document.source_id.encode())
            fingerprint.update(document.content.encode())
        self.collection_name = f"gtm_{fingerprint.hexdigest()[:20]}"
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedder.name},
        )
        if self._collection.count() == 0:
            ids: list[str] = []
            texts: list[str] = []
            metadata: list[dict[str, str | int]] = []
            for document in documents:
                for index, text in enumerate(chunk_brief(document)):
                    passage_id = f"{document.source_id}#chunk-{index + 1}"
                    ids.append(hashlib.sha256(passage_id.encode()).hexdigest())
                    texts.append(text)
                    metadata.append(
                        {
                            "passage_id": passage_id,
                            "source_id": document.source_id,
                            "chunk_index": index,
                        }
                    )
            if texts:
                self._collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadata,
                    embeddings=self._embedder.encode(texts),
                )

    def search(self, query: str, *, limit: int = 5, min_score: float = 0.10) -> list[EvidencePassage]:
        if not query.strip() or self._collection.count() == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=self._embedder.encode([query]),
                n_results=min(limit, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            passages: list[EvidencePassage] = []
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            ):
                score = max(0.0, 1.0 - float(distance))
                if score < min_score:
                    continue
                passages.append(
                    EvidencePassage(
                        passage_id=str(metadata["passage_id"]),
                        source_id=str(metadata["source_id"]),
                        text=str(text),
                        score=round(score, 6),
                        chunk_index=int(metadata["chunk_index"]),
                    )
                )
            return passages
        except Exception as exc:  # safe, explicit fallback for local model/store failures
            self.mode = "lexical_fallback"
            self.error = str(exc)
            results = [item for retriever in self._lexical for item in retriever.search(query, limit=limit)]
            return sorted(results, key=lambda item: item.score, reverse=True)[:limit]
