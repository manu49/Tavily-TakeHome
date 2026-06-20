"""
portfolio_tool.py — the `analyze_portfolio` LangChain tool.

This is the agent-facing wrapper around the deterministic quant engine. It is the heavy
tier (imports quant/marketdata, hence numpy/pandas/scipy/yfinance from the `portfolio`
extra) and is imported lazily by tavily_maxer.build_agent only when portfolio analysis is
enabled, so the research-only path and the Vercel deploy never pay for it.

Flow, mirroring TavilySearchWithRegistry:
  tickers + weights ──▶ marketdata.get_price_history ──▶ quant.compute_portfolio_stats
                    ──▶ MetricRegistry (code-owned [metric:k] ids) ──▶ labeled text to model

The model receives metrics pre-labeled with stable ids and can only *reference* them; it
never sees a path to compute or alter a number itself.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

import marketdata
import quant
from artifacts import ChartRegistry, MetricRegistry
from portfolio import Portfolio


class PortfolioAnalysisInput(BaseModel):
    tickers: List[str] = Field(description="Ticker symbols in the portfolio, e.g. ['AAPL','MSFT'].")
    weights: Optional[List[float]] = Field(
        default=None,
        description="Portfolio weights parallel to `tickers`. If omitted, equal-weighted. "
        "Need not sum to 1; they are normalized.",
    )
    period: str = Field(default="5y", description="Lookback window, e.g. '1y', '5y', '10y'.")
    risk_free_rate: float = Field(default=0.04, description="Annual risk-free rate for Sharpe/Sortino.")
    var_confidence: float = Field(default=0.95, description="VaR/CVaR confidence level, e.g. 0.95 or 0.99.")
    var_horizon_days: int = Field(default=1, description="VaR/CVaR horizon in trading days.")
    benchmark_ticker: Optional[str] = Field(
        default=None, description="Benchmark for beta, e.g. 'SPY'. Omit to skip beta."
    )
    include_charts: bool = Field(
        default=True,
        description="Also build interactive charts (performance, drawdown, return "
        "distribution, correlation, allocation) referenceable as [chart:j].",
    )


class PortfolioAnalysisTool(BaseTool):
    """Compute risk/return metrics for a portfolio over real price history."""

    name: str = "analyze_portfolio"
    description: str = (
        "Compute quantitative risk/return metrics (annualized return & volatility, Sharpe, "
        "Sortino, max drawdown, beta, and Value-at-Risk / CVaR by historical, parametric, and "
        "Monte-Carlo methods) for a portfolio of tickers and weights over real historical "
        "prices. Returns metrics pre-labeled with stable [metric:k] ids to cite. Use this for "
        "any quantitative portfolio question instead of computing numbers yourself."
    )
    args_schema: type[BaseModel] = PortfolioAnalysisInput

    _registry: MetricRegistry = PrivateAttr(default_factory=MetricRegistry)
    _chart_registry: ChartRegistry = PrivateAttr(default_factory=ChartRegistry)
    _price_source: Any = PrivateAttr(default=None)

    @property
    def registry(self) -> MetricRegistry:
        return self._registry

    @property
    def chart_registry(self) -> ChartRegistry:
        return self._chart_registry

    def set_price_source(self, source: Any) -> None:
        """Inject a marketdata.PriceSource (used by tests to stay offline)."""
        self._price_source = source

    def _run(
        self,
        tickers: List[str],
        weights: Optional[List[float]] = None,
        period: str = "5y",
        risk_free_rate: float = 0.04,
        var_confidence: float = 0.95,
        var_horizon_days: int = 1,
        benchmark_ticker: Optional[str] = None,
        include_charts: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        norm_tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not norm_tickers:
            return "ERROR: no tickers supplied."
        if weights is not None and len(weights) != len(norm_tickers):
            return (
                f"ERROR: got {len(weights)} weights for {len(norm_tickers)} tickers; "
                "they must be parallel or weights omitted for equal weighting."
            )

        fetch_list = list(norm_tickers)
        bench = benchmark_ticker.strip().upper() if benchmark_ticker else None
        if bench and bench not in fetch_list:
            fetch_list.append(bench)

        try:
            history = marketdata.get_price_history(
                fetch_list, period=period, interval="1d", source=self._price_source
            )
        except Exception as exc:  # data layer issues shouldn't kill the agent run
            return f"ERROR: could not fetch price history: {type(exc).__name__}: {exc}"

        available = [t for t in norm_tickers if t in history.tickers]
        if not available:
            return f"ERROR: no price data for any of {norm_tickers} (missing: {history.missing})."

        # Build weights over the tickers that actually have data, then renormalize.
        if weights is None:
            raw = {t: 1.0 for t in available}
        else:
            wmap = dict(zip(norm_tickers, weights))
            raw = {t: float(wmap[t]) for t in available}
        total = sum(raw.values())
        if total <= 0:
            return "ERROR: portfolio weights sum to zero; cannot analyze."
        weight_map = {t: w / total for t, w in raw.items()}

        benchmark_returns = None
        if bench and bench in history.tickers:
            benchmark_returns = quant.simple_returns(history.prices[bench])

        ppy = quant.PERIODS_PER_YEAR.get(history.interval, quant.TRADING_DAYS)
        stats = quant.compute_portfolio_stats(
            history.prices[available],
            weight_map,
            risk_free_rate=risk_free_rate,
            periods_per_year=ppy,
            var_confidence=var_confidence,
            var_horizon=var_horizon_days,
            benchmark_returns=benchmark_returns,
        )

        records = self._registry.register_portfolio_stats(stats)

        chart_records: list = []
        if include_charts:
            try:
                import charts  # lazy: only needed when charts are requested

                chart_records = charts.build_portfolio_charts(
                    history.prices[available], weight_map, stats, self._chart_registry
                )
            except Exception as exc:  # charts are a nice-to-have; never fail the analysis
                chart_records = []
                if run_manager is not None:
                    run_manager.on_text(f"(charts skipped: {type(exc).__name__}: {exc})")

        return self._format(history, weight_map, records, chart_records)

    @staticmethod
    def _format(
        history: "marketdata.PriceHistory", weights: dict, records: list, charts: list
    ) -> str:
        lines: List[str] = []
        alloc = ", ".join(f"{t} {w * 100:.1f}%" for t, w in weights.items())
        lines.append(f"Portfolio: {alloc}")
        lines.append(
            f"Data: {history.source}, {history.vintage()['rows']} obs, period {history.period}, "
            f"as of {history.as_of.date()}."
        )
        if history.missing:
            lines.append(f"(no data for, excluded: {', '.join(history.missing)})")
        lines.append("")
        lines.append("Computed metrics (cite by id as [metric:k]):")
        for r in records:
            lines.append(f"  [metric:{r.id}] {r.name} = {r.formatted()} — {r.definition}")
        chart_clause = ""
        if charts:
            lines.append("")
            lines.append("Available charts (reference relevant ones as [chart:j]):")
            for c in charts:
                lines.append(f"  [chart:{c.id}] {c.title} — {c.caption}")
            chart_clause = (
                " and embed relevant [chart:j] markers where a visual helps (list them in "
                "referenced_chart_ids)"
            )
        lines.append("")
        lines.append(
            "Cite every quantitative claim with its [metric:k] id (list them in "
            f"referenced_metric_ids){chart_clause}. Do not invent or recompute any value."
        )
        return "\n".join(lines)


def build_portfolio_tool(price_source: Any = None) -> PortfolioAnalysisTool:
    """Factory: fresh tool with an empty MetricRegistry (and optional injected price source)."""
    tool = PortfolioAnalysisTool()
    if price_source is not None:
        tool.set_price_source(price_source)
    return tool


class _NoArgs(BaseModel):
    pass


class GetPortfolioTool(BaseTool):
    """Expose the user's uploaded portfolio so the agent can analyze "my portfolio" without
    the holdings being retyped. Returns tickers and weights (or quantities); the agent then
    passes them to analyze_portfolio."""

    name: str = "get_uploaded_portfolio"
    description: str = (
        "Return the user's uploaded portfolio holdings (tickers and weights/quantities). "
        "Call this first for any question about 'my portfolio' or 'my holdings', then pass "
        "the tickers and weights to analyze_portfolio."
    )
    args_schema: type[BaseModel] = _NoArgs

    _portfolio: Optional[Portfolio] = PrivateAttr(default=None)

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: Any) -> str:
        p = self._portfolio
        if p is None or not p.holdings:
            return "No portfolio has been uploaded for this session."
        lines = [f"Uploaded portfolio: {p.name} (from {p.source_filename or 'upload'}), {p.currency}."]
        weights = p.normalized_weights()
        lines.append("Holdings:")
        for h in p.holdings:
            if weights is not None:
                lines.append(f"  {h.ticker}: weight {weights[h.ticker] * 100:.1f}%")
            elif h.quantity is not None:
                lines.append(f"  {h.ticker}: quantity {h.quantity:g}")
            else:
                lines.append(f"  {h.ticker}")
        if weights is None and p.has_quantities():
            lines.append("(weights to be derived from quantities and live prices by analyze_portfolio)")
        lines.append(
            "\nNow call analyze_portfolio with these tickers"
            + (" and weights" if weights is not None else "")
            + " to compute risk/return metrics and charts."
        )
        return "\n".join(lines)


def build_get_portfolio_tool(portfolio: Portfolio) -> GetPortfolioTool:
    tool = GetPortfolioTool()
    tool._portfolio = portfolio
    return tool
