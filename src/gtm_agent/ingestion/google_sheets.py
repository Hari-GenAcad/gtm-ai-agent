from __future__ import annotations

from collections.abc import Callable
import re
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from gtm_agent.models.schemas import ProductBrief

from .loader import load_bytes


MAX_SHEET_BYTES = 5 * 1024 * 1024
_SHEET_PATH = re.compile(r"^/spreadsheets/d/([A-Za-z0-9_-]+)(?:/.*)?$")


def google_sheet_export_url(url: str) -> tuple[str, str]:
    """Validate a public Google Sheets URL and return its CSV export URL and sheet id."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.hostname != "docs.google.com":
        raise ValueError("only HTTPS docs.google.com spreadsheet URLs are supported")
    match = _SHEET_PATH.fullmatch(parsed.path)
    if not match:
        raise ValueError("invalid Google Sheets URL")
    sheet_id = match.group(1)
    query = parse_qs(parsed.query)
    fragment = parse_qs(parsed.fragment)
    gid = (query.get("gid") or fragment.get("gid") or [None])[0]
    if gid is not None and not gid.isdigit():
        raise ValueError("Google Sheets gid must be numeric")
    parameters = {"format": "csv"}
    if gid is not None:
        parameters["gid"] = gid
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?{urlencode(parameters)}"
    return export_url, sheet_id


def _download_csv(url: str, *, timeout: float = 15, max_bytes: int = MAX_SHEET_BYTES) -> bytes:
    request = Request(url, headers={"User-Agent": "gtm-content-agent/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - host is strictly allowlisted above
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > max_bytes:
                raise ValueError("Google Sheet export exceeds the 5 MB ingestion limit")
            data = response.read(max_bytes + 1)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "unable to download the Google Sheet; confirm it is publicly readable"
        ) from exc
    if len(data) > max_bytes:
        raise ValueError("Google Sheet export exceeds the 5 MB ingestion limit")
    return data


def load_public_google_sheet(
    url: str,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> list[ProductBrief]:
    """Load a publicly readable Google Sheet as attributed CSV evidence."""
    export_url, sheet_id = google_sheet_export_url(url)
    data = (fetcher or _download_csv)(export_url)
    documents = load_bytes(f"google-sheet-{sheet_id}.csv", data)
    return [
        document.model_copy(
            update={
                "title": f"Google Sheet {sheet_id}",
                "source_id": url.strip(),
            }
        )
        for document in documents
    ]
