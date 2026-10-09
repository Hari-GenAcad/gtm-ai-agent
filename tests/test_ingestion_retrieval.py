from pathlib import Path
from io import BytesIO
import zipfile

import pytest
from openpyxl import Workbook
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from gtm_agent.ingestion import (
    google_sheet_export_url,
    load_bytes,
    load_document,
    load_documents,
    load_public_google_sheet,
    merge_documents,
    normalize_text,
)
from gtm_agent.retrieval import LexicalRetriever, analyze_evidence_quality, chunk_brief


def test_normalize_preserves_paragraphs() -> None:
    assert normalize_text(" One  line \r\n\r\n Two\tline \n") == "One line\n\nTwo line"


def test_load_markdown_tracks_filename(tmp_path: Path) -> None:
    path = tmp_path / "approved-brief.md"
    path.write_text("# Product\n\nUseful facts", encoding="utf-8")
    brief = load_document(path)
    assert brief.source_id == "approved-brief.md"
    assert brief.media_type == "markdown"
    assert "Useful facts" in brief.content


def test_unsupported_document_rejected(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    path.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_document(path)


def test_chunk_and_retrieve_with_attribution(brief) -> None:
    assert chunk_brief(brief)
    results = LexicalRetriever(brief).search("revenue campaign pricing")
    assert results
    assert all(item.source_id == "brief.md" for item in results)
    assert all(item.passage_id.startswith("brief.md#chunk-") for item in results)


def test_irrelevant_query_returns_no_evidence(brief) -> None:
    assert LexicalRetriever(brief).search("xylophone quasar nebula") == []


def test_evidence_quality_flags_contradictions_and_guarantees() -> None:
    issues = analyze_evidence_quality(
        "Launch date: May 1\nLaunch date: June 1\nThe world's best product offers guaranteed results."
    )
    assert any("Contradictory" in issue for issue in issues)
    assert any("guarantee" in issue.lower() or "world's best" in issue.lower() for issue in issues)


def test_google_sheet_csv_export_is_normalized_with_headers() -> None:
    documents = load_bytes(
        "launch-calendar.csv",
        b"Product,Launch Date,Audience\nLaunchPad,2026-11-01,Revenue leaders\n",
    )
    assert documents[0].media_type == "spreadsheet"
    assert "Product: LaunchPad" in documents[0].content
    assert "Launch Date: 2026-11-01" in documents[0].content


def test_google_sheet_xlsx_export_preserves_sheet_and_rows() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Launches"
    sheet.append(["Product", "Date", "Message"])
    sheet.append(["LaunchPad", "2026-11-01", "Grounded campaigns"])
    buffer = BytesIO()
    workbook.save(buffer)
    document = load_bytes("calendar.xlsx", buffer.getvalue())[0]
    assert document.media_type == "spreadsheet"
    assert "Sheet: Launches" in document.content
    assert "Message: Grounded campaigns" in document.content


def test_public_google_sheet_builds_export_url_and_preserves_attribution() -> None:
    source_url = "https://docs.google.com/spreadsheets/d/sheet_ABC-123/edit#gid=42"
    requested: list[str] = []

    def fetcher(url: str) -> bytes:
        requested.append(url)
        return b"Product,Launch Date\nLaunchPad,2026-11-01\n"

    documents = load_public_google_sheet(source_url, fetcher=fetcher)

    assert requested == [
        "https://docs.google.com/spreadsheets/d/sheet_ABC-123/export?format=csv&gid=42"
    ]
    assert documents[0].source_id == source_url
    assert documents[0].media_type == "spreadsheet"
    assert "Launch Date: 2026-11-01" in documents[0].content


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.google.com/spreadsheets/d/abc/edit",
        "https://evil.example/spreadsheets/d/abc/edit",
        "https://docs.google.com/document/d/abc/edit",
        "https://docs.google.com/spreadsheets/d/abc/edit#gid=not-a-number",
    ],
)
def test_public_google_sheet_rejects_unapproved_or_invalid_urls(url: str) -> None:
    with pytest.raises(ValueError):
        google_sheet_export_url(url)


def test_notion_zip_expands_markdown_and_csv_with_source_ids() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Products/LaunchPad.md", "LaunchPad helps revenue teams.")
        archive.writestr("Campaigns.csv", "Name,Tone\nLaunch,Practical\n")
        archive.writestr("ignored.png", b"not an image")
    documents = load_bytes("notion-export.zip", buffer.getvalue())
    assert len(documents) == 2
    assert {item.media_type for item in documents} == {"markdown", "spreadsheet"}
    assert all(item.source_id.startswith("notion-export.zip:") for item in documents)


def test_pdf_ingestion_extracts_text() -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (LaunchPad event on November 1) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    document = load_bytes("event.pdf", buffer.getvalue())[0]
    assert document.media_type == "pdf"
    assert "LaunchPad event on November 1" in document.content


def test_multiple_documents_merge_without_losing_source_markers(tmp_path: Path) -> None:
    first = tmp_path / "product.md"
    second = tmp_path / "calendar.csv"
    first.write_text("LaunchPad product specifications", encoding="utf-8")
    second.write_text("Event,Date\nLaunch,2026-11-01", encoding="utf-8")
    documents = load_documents([first, second])
    merged = merge_documents(documents)
    assert len(documents) == 2
    assert "[Source: product.md]" in merged.content
    assert "[Source: calendar.csv]" in merged.content
