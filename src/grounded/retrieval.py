"""Two-stage retrieval: Tavily Search for breadth, Tavily Extract for depth.

The starter agent calls ``TavilySearch`` once and reasons over ~200-character
snippets. That is too thin to *verify* anything. Here we:

  1. run each planned sub-query through Search (routed by topic/time range),
  2. deduplicate and rank the candidates across all sub-queries,
  3. Extract the full text of the best sources.

Stage 3 is what makes grounding possible: you cannot check a claim against a
snippet, only against the real passage. Steps 1 and 2 are pure functions and
covered by unit tests; only the Tavily calls touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

from opentelemetry import trace
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings
from .schemas import Source, SubQuery

tracer = trace.get_tracer("grounded.retrieval")


@dataclass
class Candidate:
    """A search hit before extraction."""

    url: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    query: str = ""


def _normalize_url(url: str) -> str:
    """Canonicalize a URL for dedup: drop scheme casing, fragments, trailing /."""
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url.strip()
    netloc = p.netloc.lower()
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), netloc, path, "", p.query, ""))


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def _search_one(subquery: SubQuery, settings: Settings) -> list[Candidate]:
    """Run a single Tavily search, routed to the right topic/time range."""
    from langchain_tavily import TavilySearch

    kwargs: dict = {
        "max_results": settings.results_per_query,
        "topic": subquery.topic.value,
        "search_depth": settings.search_depth,
    }
    if subquery.time_range:
        kwargs["time_range"] = subquery.time_range

    with tracer.start_as_current_span("tavily.search") as span:
        span.set_attribute("tavily.query", subquery.query)
        span.set_attribute("tavily.topic", subquery.topic.value)
        span.set_attribute("tavily.search_depth", settings.search_depth)
        payload = TavilySearch(**kwargs).invoke({"query": subquery.query})
        results = _coerce_results(payload)
        span.set_attribute("tavily.result_count", len(results))

    return [
        Candidate(
            url=r.get("url", ""),
            title=r.get("title", "") or "",
            snippet=" ".join((r.get("content") or "").split()),
            score=float(r.get("score") or 0.0),
            query=subquery.query,
        )
        for r in results
        if r.get("url")
    ]


def _coerce_results(payload) -> list[dict]:
    """Tavily tools usually return a dict; tolerate a JSON string too."""
    if isinstance(payload, str):
        import json

        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        return payload.get("results", []) or []
    return []


def dedupe_and_rank(candidates: list[Candidate], settings: Settings) -> list[Candidate]:
    """Merge duplicate URLs, cap per-domain, and keep the top-scoring sources.

    Pure function — no network. Domain capping keeps one loud site from
    crowding out corroborating sources, which matters for cross-checking.
    """
    best: dict[str, Candidate] = {}
    for c in candidates:
        if not c.url:
            continue
        key = _normalize_url(c.url)
        if key not in best or c.score > best[key].score:
            best[key] = c

    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)

    selected: list[Candidate] = []
    per_domain: dict[str, int] = {}
    for c in ranked:
        d = _domain(c.url)
        if per_domain.get(d, 0) >= settings.max_per_domain:
            continue
        per_domain[d] = per_domain.get(d, 0) + 1
        selected.append(c)
        if len(selected) >= settings.sources_to_extract:
            break
    return selected


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def _extract(urls: list[str], settings: Settings) -> dict[str, str]:
    """Extract full page text for a batch of URLs -> {url: content}."""
    from langchain_tavily import TavilyExtract

    with tracer.start_as_current_span("tavily.extract") as span:
        span.set_attribute("tavily.url_count", len(urls))
        payload = TavilyExtract(
            extract_depth=settings.extract_depth, format="markdown"
        ).invoke({"urls": urls})

    results = payload.get("results", []) if isinstance(payload, dict) else []
    span_failed = payload.get("failed_results", []) if isinstance(payload, dict) else []
    out: dict[str, str] = {}
    for r in results:
        url = r.get("url")
        content = r.get("raw_content") or r.get("content") or ""
        if url and content:
            out[url] = content
    if span_failed:
        trace.get_current_span().set_attribute("tavily.failed_count", len(span_failed))
    return out


def gather_sources(subqueries: list[SubQuery], settings: Settings) -> list[Source]:
    """Run the full search -> dedupe -> extract path and return numbered Sources."""
    with tracer.start_as_current_span("retrieval.gather") as span:
        candidates: list[Candidate] = []
        for sq in subqueries:
            candidates.extend(_search_one(sq, settings))
        span.set_attribute("retrieval.candidate_count", len(candidates))

        selected = dedupe_and_rank(candidates, settings)
        span.set_attribute("retrieval.selected_count", len(selected))

        extracted = _extract([c.url for c in selected], settings) if selected else {}

        sources: list[Source] = []
        for i, c in enumerate(selected, start=1):
            content = extracted.get(c.url, "")
            if not content:
                continue  # extraction failed for this URL; drop it
            sources.append(
                Source(
                    id=f"S{i}",
                    url=c.url,
                    title=c.title,
                    content=content,
                    score=c.score,
                )
            )
        span.set_attribute("retrieval.source_count", len(sources))
        return sources
