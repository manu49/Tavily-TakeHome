"""
portfolio.py — parse an uploaded portfolio file into a validated Portfolio, and persist it.

Portfolio managers upload holdings as CSV, Excel, or a plain text list; this module turns
any of those into a clean `Portfolio` (tickers + weights/quantities) with a validation
report the UI can show before any analysis runs. Parsing is deterministic and offline-
testable (it takes bytes, not a live upload).

It also provides `PortfolioStore`: a simple file-backed store with a 30-day TTL, matching
the typical month-long holding period — an uploaded portfolio is retrievable by id for a
month, then expires. (Serverless filesystems are ephemeral, so on a container host this
points at a persistent volume; the store is storage-agnostic via `store_dir`.)

Heavy tier: pandas/openpyxl are imported lazily inside the parser, so importing the
Portfolio model or the store does not require them.
"""

from __future__ import annotations

import io
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

THIRTY_DAYS_SECONDS = 30 * 24 * 3600

# Flexible, case-insensitive header synonyms for tabular uploads.
_TICKER_KEYS = {"ticker", "tickers", "symbol", "symbols", "stock", "security", "asset"}
_WEIGHT_KEYS = {"weight", "weights", "allocation", "alloc", "%", "pct", "percent", "weight(%)", "weight %"}
_QUANTITY_KEYS = {"quantity", "qty", "shares", "units", "position"}
# Market value / money columns -> weights are derived as value / total.
_VALUE_KEYS = {
    "value", "marketvalue", "dollaramount", "dollars", "dollar", "notional",
    "positionvalue", "mv", "mktval", "usd", "exposure", "amount", "marketval",
}
_COST_KEYS = {"cost", "costbasis", "avgcost", "purchaseprice", "avgprice"}

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@dataclass
class Holding:
    ticker: str
    weight: Optional[float] = None
    quantity: Optional[float] = None
    cost_basis: Optional[float] = None
    value: Optional[float] = None       # market/dollar value, if given


@dataclass
class Portfolio:
    holdings: List[Holding]
    name: str = "Uploaded portfolio"
    currency: str = "USD"
    source_filename: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def tickers(self) -> List[str]:
        return [h.ticker for h in self.holdings]

    def has_weights(self) -> bool:
        return bool(self.holdings) and all(h.weight is not None for h in self.holdings)

    def has_quantities(self) -> bool:
        return bool(self.holdings) and all(h.quantity is not None for h in self.holdings)

    def normalized_weights(self) -> Optional[Dict[str, float]]:
        """Weights summing to 1 if every holding has a weight, else None (caller derives
        weights from quantities × live prices at analysis time)."""
        if not self.has_weights():
            return None
        total = sum(h.weight for h in self.holdings)
        if total <= 0:
            return None
        return {h.ticker: h.weight / total for h in self.holdings}

    def to_dict(self) -> dict:
        return {
            "holdings": [asdict(h) for h in self.holdings],
            "name": self.name,
            "currency": self.currency,
            "source_filename": self.source_filename,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        return cls(
            holdings=[Holding(**h) for h in d.get("holdings", [])],
            name=d.get("name", "Uploaded portfolio"),
            currency=d.get("currency", "USD"),
            source_filename=d.get("source_filename"),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class ParsedPortfolio:
    portfolio: Optional[Portfolio]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.portfolio is not None and not self.errors

    def report(self) -> dict:
        return {
            "ok": self.ok,
            "warnings": self.warnings,
            "errors": self.errors,
            "portfolio": self.portfolio.to_dict() if self.portfolio else None,
        }


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

def _norm_header(h: str) -> str:
    return re.sub(r"[\s_]+", "", str(h).strip().lower())


def _coerce_float(raw) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "").replace("$", "")
    is_pct = s.endswith("%")
    s = s.rstrip("%").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100.0 if is_pct else v


def _clean_ticker(raw) -> Optional[str]:
    if raw is None:
        return None
    t = str(raw).strip().upper()
    return t if _TICKER_RE.match(t) else None


def _build_portfolio_from_rows(
    rows: List[dict], filename: str, warnings: List[str], errors: List[str]
) -> Optional[Portfolio]:
    """rows: list of {ticker, weight?, quantity?, cost_basis?, _weight_was_pct?}."""
    merged: Dict[str, Holding] = {}
    for r in rows:
        ticker = r.get("ticker")
        if not ticker:
            continue
        if ticker in merged:
            warnings.append(f"Duplicate ticker {ticker}; merged.")
            existing = merged[ticker]
            for fld in ("weight", "quantity", "value"):
                if r.get(fld) is not None:
                    setattr(existing, fld, (getattr(existing, fld) or 0.0) + r[fld])
        else:
            merged[ticker] = Holding(
                ticker=ticker, weight=r.get("weight"), quantity=r.get("quantity"),
                cost_basis=r.get("cost_basis"), value=r.get("value"),
            )

    holdings = list(merged.values())
    if not holdings:
        errors.append("No valid holdings found (need at least a ticker column/list).")
        return None

    have_weight = [h for h in holdings if h.weight is not None]
    have_value = [h for h in holdings if h.value is not None]
    have_qty = [h for h in holdings if h.quantity is not None]

    if have_weight and len(have_weight) == len(holdings):
        raw_total = sum(h.weight for h in holdings)
        # Heuristic: weights like 60/40 (sum ~100) are percentages -> scale to fractions.
        if raw_total > 1.5 and 90.0 <= raw_total <= 110.0:
            for h in holdings:
                h.weight = h.weight / 100.0
            raw_total = sum(h.weight for h in holdings)
        if abs(raw_total - 1.0) > 0.02:
            warnings.append(
                f"Weights sum to {raw_total:.3f}, not 1.0; they will be normalized."
            )
    elif have_value and len(have_value) == len(holdings):
        # Market values -> weights proportional to dollar amount.
        total_value = sum(h.value for h in holdings)
        if total_value <= 0:
            warnings.append("Market values are non-positive; assuming equal weight.")
            for h in holdings:
                h.weight = 1.0 / len(holdings)
        else:
            for h in holdings:
                h.weight = h.value / total_value
            warnings.append("Weights derived from the market-value/dollar-amount column.")
    elif have_qty and len(have_qty) == len(holdings):
        warnings.append("No weights given; weights will be derived from quantities and live prices.")
    else:
        warnings.append("No usable weights, values, or quantities; assuming equal weight.")
        for h in holdings:
            h.weight = 1.0 / len(holdings)

    return Portfolio(holdings=holdings, source_filename=filename)


def _parse_tabular(content: bytes, filename: str, warnings, errors) -> List[dict]:
    import pandas as pd  # lazy heavy import

    buffer = io.BytesIO(content)
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    else:
        # Auto-detect the delimiter (comma, tab, semicolon) so tab-separated exports work;
        # fall back to a plain comma read for single-column files the sniffer can't handle.
        try:
            df = pd.read_csv(buffer, sep=None, engine="python")
        except Exception:
            buffer.seek(0)
            df = pd.read_csv(buffer)

    if df.empty:
        return []

    colmap: Dict[str, str] = {}
    for col in df.columns:
        key = _norm_header(col)
        if key in _TICKER_KEYS:
            colmap[col] = "ticker"
        elif key in _WEIGHT_KEYS:
            colmap[col] = "weight"
        elif key in _VALUE_KEYS:
            colmap[col] = "value"
        elif key in _QUANTITY_KEYS:
            colmap[col] = "quantity"
        elif key in _COST_KEYS:
            colmap[col] = "cost_basis"

    if "ticker" not in colmap.values():
        # Fall back to the first column as tickers.
        first = df.columns[0]
        colmap[first] = "ticker"
        warnings.append(f"No recognized ticker column; using '{first}'.")

    rows: List[dict] = []
    for _, raw_row in df.iterrows():
        row: Dict[str, object] = {}
        for col, role in colmap.items():
            value = raw_row[col]
            if role == "ticker":
                row["ticker"] = _clean_ticker(value)
            else:
                row[role] = _coerce_float(value)
        if row.get("ticker"):
            rows.append(row)
        elif row:  # had a row but ticker didn't validate
            bad = raw_row[[c for c, r in colmap.items() if r == "ticker"][0]]
            if str(bad).strip():
                warnings.append(f"Skipped unrecognized ticker {str(bad).strip()!r}.")
    return rows


def _parse_text(content: bytes, warnings, errors) -> List[dict]:
    """Plain-text list: one holding per line, e.g. 'AAPL 60%', 'MSFT, 40', 'GOOG 0.2'."""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        errors.append("Could not decode text file as UTF-8.")
        return []

    rows: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in re.split(r"[\s,;]+", line) if p]
        ticker = _clean_ticker(parts[0]) if parts else None
        if not ticker:
            if parts:
                warnings.append(f"Skipped line (no valid ticker): {line!r}")
            continue
        row: Dict[str, object] = {"ticker": ticker}
        if len(parts) > 1:
            # Find the first token after the ticker that parses as a number. A multi-token
            # line with no number is almost certainly prose (e.g. "not a ticker line"), not a
            # holding -- skip it rather than treat the first word as a ticker.
            value_token = next((p for p in parts[1:] if _coerce_float(p) is not None), None)
            if value_token is None:
                warnings.append(f"Skipped line (looks like prose, no value): {line!r}")
                continue
            val = _coerce_float(value_token)
            # A value with %, or <= 1, reads as a weight; a larger bare number as a quantity.
            if "%" in value_token or val <= 1.0:
                row["weight"] = val
            else:
                row["quantity"] = val
        rows.append(row)
    return rows


def parse_portfolio(content: bytes, filename: str = "portfolio.csv") -> ParsedPortfolio:
    """Parse uploaded bytes into a validated Portfolio. Never raises on bad data: problems
    are reported in `warnings`/`errors` so the caller can show a report and decide."""
    warnings: List[str] = []
    errors: List[str] = []

    if not content or not content.strip():
        return ParsedPortfolio(None, warnings, ["Uploaded file is empty."])

    name = filename.lower()
    try:
        if name.endswith((".csv", ".xlsx", ".xls")):
            rows = _parse_tabular(content, filename, warnings, errors)
        elif name.endswith((".txt", ".text")) or "." not in name:
            rows = _parse_text(content, warnings, errors)
        else:
            # Unknown extension: try tabular, then fall back to text.
            try:
                rows = _parse_tabular(content, filename, warnings, errors)
            except Exception:
                rows = _parse_text(content, warnings, errors)
    except Exception as exc:
        return ParsedPortfolio(None, warnings, [f"Could not parse file: {type(exc).__name__}: {exc}"])

    portfolio = _build_portfolio_from_rows(rows, filename, warnings, errors)
    return ParsedPortfolio(portfolio, warnings, errors)


# --------------------------------------------------------------------------------------
# Persistence (30-day TTL)
# --------------------------------------------------------------------------------------

class PortfolioStore:
    """File-backed portfolio store with a 30-day TTL (the typical holding period). Each
    portfolio is one JSON file named by id; entries older than the TTL are treated as
    expired (and purged on access)."""

    def __init__(self, store_dir: str | Path = "uploads/portfolios", ttl_seconds: int = THIRTY_DAYS_SECONDS):
        self.store_dir = Path(store_dir)
        self.ttl_seconds = ttl_seconds

    def _path(self, portfolio_id: str) -> Path:
        return self.store_dir / f"{portfolio_id}.json"

    def save(self, portfolio: Portfolio) -> str:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        portfolio_id = uuid.uuid4().hex
        self._path(portfolio_id).write_text(json.dumps(portfolio.to_dict()), encoding="utf-8")
        return portfolio_id

    def load(self, portfolio_id: str) -> Optional[Portfolio]:
        path = self._path(portfolio_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - data.get("created_at", 0) > self.ttl_seconds:
            path.unlink(missing_ok=True)  # expired: purge
            return None
        return Portfolio.from_dict(data)

    def purge_expired(self) -> int:
        """Delete all expired entries; returns the count removed."""
        if not self.store_dir.exists():
            return 0
        removed = 0
        now = time.time()
        for path in self.store_dir.glob("*.json"):
            try:
                created = json.loads(path.read_text(encoding="utf-8")).get("created_at", 0)
            except (json.JSONDecodeError, OSError):
                created = 0
            if now - created > self.ttl_seconds:
                path.unlink(missing_ok=True)
                removed += 1
        return removed
