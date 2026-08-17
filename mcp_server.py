"""
MCP server for EAGV3 Session 7.

Twelve tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    weather_forecast,
    read_file, list_dir, create_file, update_file, edit_file,
    index_document, search_knowledge

web_search:        Tavily optional, DuckDuckGo primary, Bing fallback. Max 5.
fetch_url:         Crawl4AI primary, requests/BeautifulSoup fallback.
index_document:    Sliding-window ingestion into Memory and FAISS.
search_knowledge:  Vector retrieval over indexed fact chunks.
Usage for tavily and duckduckgo is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import json
import os
import base64
import re
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings(
    "ignore",
    message="Field 'lifespan' has an incomplete definition",
)

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

import artifacts as _artifacts
import memory as _memory

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result

# Keep Crawl4AI's SQLite/cache files inside this project. This is writable in
# the assignment/demo environment and avoids dependence on a global home cache.
os.environ.setdefault(
    "CRAWL4_AI_BASE_DIRECTORY", str(Path(__file__).parent / "state")
)

load_dotenv(Path(__file__).parent / ".env")

mcp = FastMCP("eagv3-s7-server")

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    hits: list[dict] = []
    seen: set[str] = set()
    with DDGS() as ddgs:
        for backend in ("auto",):
            try:
                batch = list(
                    ddgs.text(query, max_results=max_results, backend=backend)
                )
            except Exception:
                batch = []
            for hit in batch:
                url = hit.get("href") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                hits.append(hit)
                if len(hits) >= max_results:
                    break
            if len(hits) >= max_results:
                break
    results = [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]
    if len(results) < max_results:
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(
                "https://www.bing.com/search",
                params={"q": query},
                headers={"User-Agent": "Session7Assignment/1.0 (student demo)"},
                timeout=20,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            known = {item["url"] for item in results}
            for row in soup.select("li.b_algo"):
                anchor = row.select_one("h2 a")
                if anchor is None:
                    continue
                url = anchor.get("href") or ""
                encoded = re.search(r"[?&]u=a1([^&]+)", url)
                if encoded:
                    token = encoded.group(1)
                    try:
                        url = base64.urlsafe_b64decode(
                            token + "=" * (-len(token) % 4)
                        ).decode("utf-8")
                    except Exception:
                        pass
                if not url.startswith("http") or url in known:
                    continue
                snippet_node = row.select_one("p")
                results.append(
                    {
                        "title": anchor.get_text(" ", strip=True),
                        "url": url,
                        "snippet": (
                            snippet_node.get_text(" ", strip=True)
                            if snippet_node is not None
                            else ""
                        ),
                    }
                )
                known.add(url)
                if len(results) >= max_results:
                    break
        except Exception:
            pass
    return results[:max_results]


async def _crawl4ai_fetch(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    # crawl4ai uses Rich which writes via its own captured stdout reference, so
    # contextlib.redirect_stdout doesn't catch it. Redirect at the file-descriptor
    # level — crawl4ai's banner / [FETCH] / [SCRAPE] markers would otherwise
    # corrupt the MCP stdio JSON-RPC stream.
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    crawl_error: Exception | None = None
    r = None
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            r = await crawler.arun(url=url)
    except Exception as exc:
        crawl_error = exc
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    if crawl_error is not None:
        # Some locked-down environments cannot launch Chromium. Preserve the
        # same fetch contract with an explicit, visible HTTP fallback.
        from bs4 import BeautifulSoup
        import requests

        response = requests.get(
            url,
            headers={"User-Agent": "Session7Assignment/1.0 (student demo)"},
            timeout=30,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        text = "\n".join(
            line.strip()
            for line in soup.get_text("\n").splitlines()
            if line.strip()
        )
        return {
            "status": response.status_code,
            "content_type": "text/plain",
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
            "fetcher": "httpx-fallback",
            "warning": f"Crawl4AI unavailable: {type(crawl_error).__name__}",
        }

    assert r is not None
    # r.markdown is a str subclass (StringCompatibleMarkdown) that Pydantic
    # serializes as {} because its real field is private. Pull the raw string
    # out and force a plain str so FastMCP serializes correctly.
    md = r.markdown
    raw = (
        getattr(md, "raw_markdown", None)
        or getattr(md, "fit_markdown", None)
        or md
        or r.cleaned_html
        or r.html
        or ""
    )
    text = str(raw)
    return {
        "status": int(getattr(r, "status_code", None) or 200),
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (Tavily primary, DDG fallback). Hard-capped at 5 results. Example: web_search("python asyncio tutorial", 3)."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    effective_query = query
    lower_query = query.lower()
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    if any(word in lower_query for word in ("weather", "forecast")):
        for weekday, target_number in weekdays.items():
            if weekday not in lower_query:
                continue
            today = datetime.now().date()
            days_ahead = (target_number - today.weekday()) % 7
            target = today + timedelta(days=days_ahead or 7)
            if str(target.year) not in query:
                effective_query = f"{query} {target.strftime('%B %d %Y')}"
                max_results = max(max_results, 3)
            break
    if os.environ.get("TAVILY_API_KEY") and _under_cap("tavily"):
        try:
            results = _tavily_search(effective_query, max_results)
            if results:
                _bump("tavily")
                return results
        except Exception:
            _bump("tavily", "errors")
    results = _ddg_search(effective_query, max_results)
    _bump("duckduckgo")
    return results


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> dict:
    """Fetch clean markdown from a URL via crawl4ai (headless Chromium). Example: fetch_url("https://example.com")."""
    return await _crawl4ai_fetch(url)


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Example: get_time("Asia/Kolkata")."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def weather_forecast(location: str, day: str = "Saturday") -> dict:
    """Get a daily forecast for a named location and upcoming weekday via Open-Meteo. Example: weather_forecast("Tokyo", "Saturday")."""
    import requests

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=20,
    )
    geo_response.raise_for_status()
    places = geo_response.json().get("results") or []
    if not places:
        raise ValueError(f"No coordinates found for {location!r}")
    place = places[0]
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    target_number = weekdays.get(day.lower())
    if target_number is None:
        raise ValueError("day must be a weekday name")
    today = datetime.now().date()
    days_ahead = (target_number - today.weekday()) % 7
    target = today + timedelta(days=days_ahead or 7)
    forecast_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "timezone": place["timezone"],
            "forecast_days": 16,
        },
        timeout=20,
    )
    forecast_response.raise_for_status()
    daily = forecast_response.json()["daily"]
    try:
        position = daily["time"].index(target.isoformat())
    except ValueError as exc:
        raise ValueError(f"Forecast for {target.isoformat()} is not available") from exc
    code = int(daily["weather_code"][position])
    descriptions = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
        55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
        81: "rain showers", 82: "heavy rain showers", 95: "thunderstorm",
    }
    return {
        "location": f"{place['name']}, {place.get('country', '')}".rstrip(", "),
        "day": day.capitalize(),
        "date": target.isoformat(),
        "conditions": descriptions.get(code, f"WMO weather code {code}"),
        "temperature_min_c": daily["temperature_2m_min"][position],
        "temperature_max_c": daily["temperature_2m_max"][position],
        "precipitation_probability_max_percent": daily[
            "precipitation_probability_max"
        ][position],
        "source": "Open-Meteo",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox. Example: read_file("notes.txt")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> dict:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    entries = []
    names: list[str] = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        entries.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
        names.append(child.name)
    return {"path": path, "count": len(entries), "names": names, "entries": entries}


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


def _read_for_index(path: str) -> tuple[str, str]:
    """Return text and provenance for a sandbox file or artifact handle."""
    if path.startswith("art:"):
        return _artifacts.get_bytes(path).decode("utf-8", errors="replace"), path
    sandbox_path = _safe(path)
    return sandbox_path.read_text(encoding="utf-8"), f"sandbox:{path}"


def _chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Session 7's intentionally simple sliding word window."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    stride = max(1, size - overlap)
    position = 0
    while position < len(words):
        chunks.append(" ".join(words[position : position + size]))
        if position + size >= len(words):
            break
        position += stride
    return chunks


@mcp.tool()
def index_document(path: str, chunk_size: int = 400, overlap: int = 80) -> dict:
    """Chunk a sandbox file or artifact into searchable Memory facts.

    Use this for content that must remain searchable across later turns or
    runs. For one-shot inspection of a known sandbox file, use read_file.
    """
    text, source = _read_for_index(path)
    if not text.strip():
        return {
            "path": path,
            "source": source,
            "chunks_indexed": 0,
            "warning": "empty content",
        }
    chunks = _chunk_text(text, size=chunk_size, overlap=overlap)
    run_id = f"index-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    for index, chunk in enumerate(chunks):
        preview = chunk[:120].replace("\n", " ")
        descriptor = f"[{source} chunk {index + 1}/{len(chunks)}] {preview}"
        _memory.add_fact(
            descriptor=descriptor,
            value={
                "chunk": chunk,
                "chunk_index": index,
                "total_chunks": len(chunks),
                "source": source,
            },
            source=source,
            run_id=run_id,
        )
    return {
        "path": path,
        "source": source,
        "chunks_indexed": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }


@mcp.tool()
def search_knowledge(query: str, k: int = 5) -> dict:
    """Vector search over indexed facts, returning ranked text and sources."""
    limit = max(1, min(k, 10))
    candidates = _memory.read(query, kinds=["fact"], top_k=max(50, limit * 10))
    indexed = [
        item for item in candidates if item.source.startswith(("sandbox:", "art:"))
    ]

    # arXiv page captures contain navigation and submission-history windows.
    # Prefer substantive windows and cap repeated hits from one document so a
    # cross-document question receives genuinely comparative evidence.
    noise_markers = (
        "skip to main content",
        "## submission history",
        "## access paper",
        "current browse context:",
        "full-text links:",
        "arxivlabs",
        "disable mathjax",
        "references & citations",
    )

    def _clean_view(item) -> str:
        chunk = str(item.value.get("chunk") or item.value.get("raw") or "")
        lower = chunk.lower()
        abstract_at = lower.find("abstract:")
        if abstract_at >= 0:
            start = abstract_at + len("abstract:")
            ends = [
                lower.find(marker, start)
                for marker in noise_markers
                if lower.find(marker, start) >= 0
            ]
            end = min(ends) if ends else len(chunk)
            abstract = chunk[start:end].strip()
            if len(abstract) >= 120:
                return abstract
        marker_positions = [
            lower.find(marker) for marker in noise_markers if marker in lower
        ]
        if marker_positions:
            prefix = chunk[: min(marker_positions)].strip()
            return prefix if len(prefix) >= 200 else ""
        return chunk.strip()

    cleaned = [(item, _clean_view(item)) for item in indexed]
    pool = [(item, text) for item, text in cleaned if text]
    if not pool:
        pool = cleaned
    items: list[tuple[object, str]] = []
    per_source: dict[str, int] = {}
    for item, text in pool:
        if per_source.get(item.source, 0) >= 1:
            continue
        items.append((item, text))
        per_source[item.source] = per_source.get(item.source, 0) + 1
        if len(items) >= limit:
            break
    chunks = [
        {
            "id": item.id,
            "descriptor": item.descriptor,
            "source": item.source,
            "chunk": text,
            "metadata": {
                key: value
                for key, value in item.value.items()
                if key not in ("chunk", "raw")
            },
        }
        for item, text in items
    ]
    return {"query": query, "count": len(chunks), "chunks": chunks}


if __name__ == "__main__":
    mcp.run(transport="stdio")
