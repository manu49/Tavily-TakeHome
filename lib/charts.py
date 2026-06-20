"""
charts.py — interactive Plotly figures for portfolio analysis (heavy tier).

Produces Plotly figure specs (JSON: data + layout) that the frontend renders with plotly.js
as interactive charts — hover, zoom, pan, legend toggles — no static images. Each figure is
built strictly from code-owned data (fetched price series and quant.py results), never from
numbers supplied by the model, and registered in a ChartRegistry under a stable [chart:j] id.

Like portfolio_tool, this is imported lazily (it needs plotly + pandas from the `portfolio`
extra), so the research path and the Vercel deploy never load it.

The public entry point is `build_portfolio_charts(...)`, which builds the standard set
(cumulative price, drawdown, return distribution with VaR/CVaR markers, correlation heatmap,
allocation) and registers each, returning the ChartRecords.
"""

from __future__ import annotations

import json
from typing import List, Mapping

import pandas as pd
import plotly.graph_objects as go

import quant
from artifacts import ChartRecord, ChartRegistry

# A restrained palette / template so charts look consistent in the dark UI.
_TEMPLATE = "plotly_dark"
_PAPER = "rgba(0,0,0,0)"


def figure_to_spec(fig: go.Figure) -> dict:
    """Plotly Figure -> plain JSON-serializable dict (data + layout). Goes through
    fig.to_json() so numpy arrays and datetimes are properly encoded for the browser."""
    return json.loads(fig.to_json())


def _style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template=_TEMPLATE,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def price_history_figure(prices: pd.DataFrame) -> go.Figure:
    """Each holding's price rebased to 100 at the start, so paths are comparable."""
    fig = go.Figure()
    for col in prices.columns:
        series = prices[col].dropna()
        if series.empty:
            continue
        rebased = series / series.iloc[0] * 100.0
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased.values, mode="lines", name=col))
    fig.update_yaxes(title="Value (start = 100)")
    return _style(fig, "Cumulative performance (rebased to 100)")


def drawdown_figure(wealth: pd.Series) -> go.Figure:
    """Portfolio drawdown over time as a filled area (always <= 0)."""
    dd = quant.drawdown_series(wealth) * 100.0
    fig = go.Figure(
        go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown",
                   line=dict(color="#ff6b6b"))
    )
    fig.update_yaxes(title="Drawdown (%)")
    return _style(fig, "Portfolio drawdown")


def return_distribution_figure(
    returns: pd.Series, var: float, cvar: float, confidence: float
) -> go.Figure:
    """Histogram of periodic returns with VaR and CVaR thresholds marked. VaR/CVaR are
    positive loss fractions, so they sit on the negative-return (left) tail at -var / -cvar."""
    pct = (returns * 100.0).dropna()
    fig = go.Figure(go.Histogram(x=pct.values, nbinsx=60, name="Daily returns",
                                 marker=dict(color="#4f7cff")))
    conf_label = f"{confidence * 100:.0f}%"
    fig.add_vline(x=-var * 100.0, line=dict(color="#ffce54", dash="dash"),
                  annotation_text=f"VaR {conf_label}", annotation_position="top left")
    fig.add_vline(x=-cvar * 100.0, line=dict(color="#ff6b6b", dash="dot"),
                  annotation_text=f"CVaR {conf_label}", annotation_position="bottom left")
    fig.update_xaxes(title="Return (%)")
    fig.update_yaxes(title="Frequency")
    return _style(fig, "Return distribution with tail risk")


def correlation_heatmap_figure(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=corr.round(2).values, texttemplate="%{text}", hoverongaps=False,
    ))
    return _style(fig, "Correlation of holdings")


def allocation_figure(weights: Mapping[str, float]) -> go.Figure:
    labels = list(weights.keys())
    values = [weights[k] for k in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45, textinfo="label+percent"))
    return _style(fig, "Allocation")


def build_portfolio_charts(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    stats,
    registry: ChartRegistry,
    *,
    var_method: str = "historical",
) -> List[ChartRecord]:
    """Build and register the standard portfolio chart set from code-owned data.

    Charts that need >= 2 holdings (correlation, allocation) are skipped for a single-asset
    portfolio. Returns the registered ChartRecords (ids the model may reference as
    [chart:j])."""
    rets = quant.simple_returns(prices)
    if isinstance(rets, pd.Series):
        rets = rets.to_frame()
    port_rets = quant.portfolio_returns(rets, weights)
    wealth = (1.0 + port_rets).cumprod()

    conf = float(getattr(stats, "inputs", {}).get("var_confidence", 0.95))
    var = float(getattr(stats, "var", {}).get(var_method, 0.0))
    cvar = float(getattr(stats, "cvar", {}).get(var_method, getattr(stats, "cvar", {}).get("historical", 0.0)))

    records: List[ChartRecord] = []

    def add(kind, title, caption, fig):
        records.append(registry.register(kind, title, caption, figure_to_spec(fig)))

    add("price_history", "Cumulative performance",
        "Each holding rebased to 100 at the start of the window.",
        price_history_figure(prices))
    add("drawdown", "Portfolio drawdown",
        "Peak-to-trough decline of the weighted portfolio over time.",
        drawdown_figure(wealth))
    add("return_distribution", "Return distribution with tail risk",
        f"Histogram of portfolio daily returns with {conf * 100:.0f}% VaR and CVaR marked.",
        return_distribution_figure(port_rets, var, cvar, conf))

    if rets.shape[1] >= 2:
        add("correlation", "Correlation of holdings",
            "Pairwise return correlations across the holdings.",
            correlation_heatmap_figure(quant.correlation_matrix(rets)))
        add("allocation", "Allocation",
            "Portfolio weights by holding.",
            allocation_figure(weights))

    return records
