from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from gtm_agent.models.schemas import EvidencePassage, ProductBrief


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%+-]*")
FACT_LINE_RE = re.compile(r"^(launch date|price|pricing|availability)\s*:\s*(.+)$", re.IGNORECASE)
RISKY_CLAIMS = ("world's best", "guaranteed results", "100% guaranteed", "#1 product")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def chunk_brief(brief: ProductBrief, *, max_chars: int = 700) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in brief.content.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def analyze_evidence_quality(text: str) -> list[str]:
    """Flag narrow, deterministic evidence risks that require human resolution."""
    labelled: dict[str, set[str]] = {}
    for line in text.splitlines():
        match = FACT_LINE_RE.match(line.strip())
        if match:
            labelled.setdefault(match.group(1).casefold(), set()).add(match.group(2).strip().casefold())
    issues = [
        f"Contradictory values for {key}: {', '.join(sorted(values))}."
        for key, values in labelled.items()
        if len(values) > 1
    ]
    lower = text.casefold()
    for phrase in RISKY_CLAIMS:
        if phrase in lower:
            issues.append(f"Potentially unsupported superlative or guarantee: {phrase!r}.")
    return issues


@dataclass(slots=True)
class LexicalRetriever:
    """Small local BM25 index; deliberately a lexical, not semantic, retriever."""

    brief: ProductBrief
    chunks: list[str] = field(init=False)
    _index: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.chunks = chunk_brief(self.brief)
        corpus = [tokenize(chunk) for chunk in self.chunks]
        self._index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, limit: int = 5) -> list[EvidencePassage]:
        query_tokens = tokenize(query)
        if not self._index or not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), pair[0]))
        query_set = set(query_tokens)
        matching = [
            (idx, max(0.0, float(score)))
            for idx, score in ranked
            if query_set.intersection(tokenize(self.chunks[idx]))
        ][:limit]
        return [
            EvidencePassage(
                passage_id=f"{self.brief.source_id}#chunk-{idx + 1}",
                source_id=self.brief.source_id,
                text=self.chunks[idx],
                score=round(score, 6),
                chunk_index=idx,
            )
            for idx, score in matching
        ]
