"""Tests for the pure retrieval logic (no network)."""

from grounded.config import Settings
from grounded.retrieval import Candidate, _domain, _normalize_url, dedupe_and_rank


def test_normalize_url_collapses_equivalent_urls():
    a = _normalize_url("https://Example.com/Path/")
    b = _normalize_url("https://example.com/Path")
    assert a == b == "https://example.com/Path"


def test_domain_strips_www():
    assert _domain("https://www.example.com/x") == "example.com"
    assert _domain("https://news.example.com/y") == "news.example.com"


def test_dedupe_keeps_highest_score():
    settings = Settings(sources_to_extract=10, max_per_domain=5)
    cands = [
        Candidate(url="https://a.com/1", score=0.5),
        Candidate(url="https://a.com/1/", score=0.9),  # duplicate after normalization
    ]
    out = dedupe_and_rank(cands, settings)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_dedupe_caps_per_domain_and_ranks_by_score():
    settings = Settings(sources_to_extract=10, max_per_domain=2)
    cands = [
        Candidate(url="https://a.com/1", score=0.9),
        Candidate(url="https://a.com/2", score=0.8),
        Candidate(url="https://a.com/3", score=0.7),  # 3rd from a.com -> dropped
        Candidate(url="https://b.com/1", score=0.6),
    ]
    out = dedupe_and_rank(cands, settings)
    urls = [c.url for c in out]
    assert urls == ["https://a.com/1", "https://a.com/2", "https://b.com/1"]
    assert sum(1 for c in out if _domain(c.url) == "a.com") == 2


def test_dedupe_respects_sources_to_extract_limit():
    settings = Settings(sources_to_extract=2, max_per_domain=5)
    cands = [Candidate(url=f"https://s{i}.com", score=i / 10) for i in range(5)]
    out = dedupe_and_rank(cands, settings)
    assert len(out) == 2
    assert out[0].score >= out[1].score  # ranked
