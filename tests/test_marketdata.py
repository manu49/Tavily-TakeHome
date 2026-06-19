"""Tests for marketdata.py using a fake PriceSource — no network.

These exercise normalization, missing-ticker handling, caching, and provenance. The live
yfinance path is intentionally not hit here (it's networked and non-deterministic); the
PriceSource protocol exists precisely so this logic is testable offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

import marketdata as md


class FakeSource:
    """Returns deterministic prices for known tickers; omits unknown ones. Counts fetches
    so caching can be asserted."""

    name = "fake"

    def __init__(self, known=("AAA", "BBB")):
        self.known = set(known)
        self.calls = 0

    def fetch(self, tickers, period, interval):
        self.calls += 1
        idx = pd.date_range("2021-01-01", periods=5, freq="B")
        data = {t: [10.0, 11.0, 12.0, 11.5, 12.5] for t in tickers if t in self.known}
        return pd.DataFrame(data, index=idx)


@pytest.fixture(autouse=True)
def _clear_cache():
    md.clear_cache()
    yield
    md.clear_cache()


class TestGetPriceHistory:
    def test_returns_prices_and_metadata(self):
        h = md.get_price_history(["AAA", "BBB"], source=FakeSource())
        assert list(h.prices.columns) == ["AAA", "BBB"]
        assert h.source == "fake"
        assert h.missing == []
        assert len(h.prices) == 5

    def test_records_missing_tickers(self):
        h = md.get_price_history(["AAA", "ZZZ"], source=FakeSource())
        assert h.tickers == ["AAA"]
        assert h.missing == ["ZZZ"]

    def test_tickers_are_uppercased_and_deduped(self):
        src = FakeSource()
        h = md.get_price_history(["aaa", "AAA", " bbb "], source=src)
        assert h.tickers == ["AAA", "BBB"]

    def test_empty_tickers_raise(self):
        with pytest.raises(ValueError):
            md.get_price_history(["", "  "], source=FakeSource())

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            md.get_price_history(["AAA"], interval="1h", source=FakeSource())

    def test_single_ticker(self):
        h = md.get_price_history(["AAA"], source=FakeSource())
        assert h.tickers == ["AAA"]

    def test_vintage_dict_is_audit_ready(self):
        h = md.get_price_history(["AAA"], period="1y", interval="1wk", source=FakeSource())
        v = h.vintage()
        assert v["source"] == "fake"
        assert v["period"] == "1y"
        assert v["interval"] == "1wk"
        assert v["rows"] == 5
        assert "as_of" in v


class TestCaching:
    def test_second_call_is_served_from_cache(self):
        src = FakeSource()
        md.get_price_history(["AAA"], source=src)
        md.get_price_history(["AAA"], source=src)
        assert src.calls == 1  # second call hit the cache

    def test_cache_can_be_bypassed(self):
        src = FakeSource()
        md.get_price_history(["AAA"], source=src, cache_ttl_seconds=0)
        md.get_price_history(["AAA"], source=src, cache_ttl_seconds=0)
        assert src.calls == 2

    def test_different_period_is_a_separate_cache_entry(self):
        src = FakeSource()
        md.get_price_history(["AAA"], period="1y", source=src)
        md.get_price_history(["AAA"], period="5y", source=src)
        assert src.calls == 2
