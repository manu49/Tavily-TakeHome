"""
marketdata.py — adjusted-close price history for the quant engine.

Tavily returns *text*; VaR/Sharpe/beta need numeric time series. This module fetches them
behind a small `PriceSource` protocol so the demo can use yfinance (free) while a licensed
feed (Polygon, FMP, Bloomberg) drops in later without touching the agent or `quant.py`.

Every fetch returns a `PriceHistory` that carries its **data vintage** (as-of timestamp,
source name, requested period/interval) so any metric computed downstream is reproducible
and auditable — the same "prove where it came from" ethos as the citation registry.

Design notes:
  * The agent-facing entry point is `get_price_history(...)`, which returns a tidy
    DataFrame of adjusted closes (one column per ticker) plus metadata.
  * Network access lives only in `YFinanceSource`. Everything else (normalization, caching,
    the protocol) is pure and unit-testable with a fake source — no network in tests.
  * Unknown/again delisted tickers are reported in `PriceHistory.missing` rather than raised,
    so the model can react (drop the ticker, tell the user) instead of the run dying.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Protocol, Sequence

import pandas as pd

VALID_INTERVALS = {"1d", "1wk", "1mo"}


@dataclass
class PriceHistory:
    """Adjusted-close panel plus the provenance needed to reproduce it."""

    prices: pd.DataFrame                  # index: dates, columns: tickers, values: adj close
    source: str                          # e.g. "yfinance"
    period: str                          # e.g. "5y"
    interval: str                        # e.g. "1d"
    as_of: datetime                      # when this snapshot was taken (UTC)
    missing: List[str] = field(default_factory=list)  # requested tickers with no data

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)

    def vintage(self) -> Dict[str, object]:
        """Audit dict to attach to metrics / traces."""
        return {
            "source": self.source,
            "period": self.period,
            "interval": self.interval,
            "as_of": self.as_of.isoformat(),
            "rows": int(len(self.prices)),
            "tickers": self.tickers,
            "missing": self.missing,
        }


class PriceSource(Protocol):
    """Anything that can return adjusted-close prices for tickers. Implement this to swap
    yfinance for a licensed feed."""

    name: str

    def fetch(self, tickers: Sequence[str], period: str, interval: str) -> pd.DataFrame:
        """Return a DataFrame of adjusted closes indexed by date, one column per ticker.
        Columns for tickers with no data may be omitted; callers detect them as missing."""
        ...


class YFinanceSource:
    """PriceSource backed by yfinance (auto-adjusted closes). The only networked code here."""

    name = "yfinance"

    def fetch(self, tickers: Sequence[str], period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            list(tickers),
            period=period,
            interval=interval,
            auto_adjust=True,     # 'Close' is split/dividend-adjusted
            progress=False,
            group_by="column",
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        # Single ticker -> flat columns; multiple -> MultiIndex (field, ticker).
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].copy()
            close.columns = [list(tickers)[0]]

        return close.dropna(how="all")


@dataclass
class _CacheEntry:
    history: PriceHistory
    stored_at: float


# Prices go stale (new bars print intraday/daily), so this is a *short* freshness cache to
# avoid hammering the source within a single analysis — not the month-long persistence of an
# uploaded portfolio, which is a separate concern handled at the session layer.
_CACHE: Dict[tuple, _CacheEntry] = {}
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour


def _normalize_tickers(tickers: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for t in tickers:
        key = t.strip().upper()
        if key:
            seen.setdefault(key, None)
    return list(seen)


def clear_cache() -> None:
    _CACHE.clear()


def get_price_history(
    tickers: Sequence[str],
    period: str = "5y",
    interval: str = "1d",
    *,
    source: PriceSource | None = None,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> PriceHistory:
    """Fetch adjusted-close history for `tickers`, normalized and cached.

    Tickers are upper-cased and de-duplicated (order preserved). Any requested ticker with
    no returned data is recorded in `PriceHistory.missing`. Results are cached per
    (tickers, period, interval) for `cache_ttl_seconds` to avoid duplicate fetches within a
    run; pass `cache_ttl_seconds=0` to bypass the cache.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}, got {interval!r}")

    requested = _normalize_tickers(tickers)
    if not requested:
        raise ValueError("No valid tickers supplied.")

    src = source or YFinanceSource()
    cache_key = (tuple(requested), period, interval, src.name)

    if cache_ttl_seconds > 0:
        entry = _CACHE.get(cache_key)
        if entry is not None and (time.time() - entry.stored_at) < cache_ttl_seconds:
            return entry.history

    frame = src.fetch(requested, period, interval)
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    # Keep only requested tickers that actually came back with data.
    available = [t for t in requested if t in frame.columns and frame[t].notna().any()]
    missing = [t for t in requested if t not in available]
    prices = frame[available].sort_index() if available else pd.DataFrame()

    history = PriceHistory(
        prices=prices,
        source=src.name,
        period=period,
        interval=interval,
        as_of=datetime.now(timezone.utc),
        missing=missing,
    )

    if cache_ttl_seconds > 0:
        _CACHE[cache_key] = _CacheEntry(history=history, stored_at=time.time())
    return history
