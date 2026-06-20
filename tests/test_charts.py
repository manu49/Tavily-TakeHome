"""Tests for charts.py — Plotly figure builders. Offline: synthetic prices, no network.

Asserts the figures are well-formed JSON specs (data + layout) with the expected traces and
tail-risk markers, and that build_portfolio_charts registers the right set.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import charts
import quant
from artifacts import ChartRegistry


@pytest.fixture
def prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 300)),
            "BBB": 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 300)),
        },
        index=idx,
    )


@pytest.fixture
def stats(prices):
    return quant.compute_portfolio_stats(prices, {"AAA": 0.6, "BBB": 0.4}, var_confidence=0.95)


class TestFigureToSpec:
    def test_spec_is_json_serializable_with_data_and_layout(self, prices):
        spec = charts.figure_to_spec(charts.price_history_figure(prices))
        json.dumps(spec)  # must not raise
        assert "data" in spec and "layout" in spec


class TestIndividualFigures:
    def test_price_history_has_one_trace_per_ticker(self, prices):
        fig = charts.price_history_figure(prices)
        names = {tr.name for tr in fig.data}
        assert names == {"AAA", "BBB"}

    def test_price_history_rebased_to_100(self, prices):
        fig = charts.price_history_figure(prices)
        assert fig.data[0].y[0] == pytest.approx(100.0)

    def test_return_distribution_marks_var_and_cvar(self, prices, stats):
        port_rets = quant.portfolio_returns(quant.simple_returns(prices), {"AAA": 0.6, "BBB": 0.4})
        fig = charts.return_distribution_figure(
            port_rets, stats.var["historical"], stats.cvar["historical"], 0.95
        )
        # two vertical lines (VaR, CVaR) live as layout shapes; both on the left/loss side
        xs = [s["x0"] for s in fig.layout.shapes]
        assert len(fig.layout.shapes) == 2
        assert all(x < 0 for x in xs)

    def test_correlation_heatmap_is_square(self, prices):
        corr = quant.correlation_matrix(quant.simple_returns(prices))
        fig = charts.correlation_heatmap_figure(corr)
        assert np.array(fig.data[0].z).shape == (2, 2)

    def test_allocation_pie_labels(self):
        fig = charts.allocation_figure({"AAA": 0.6, "BBB": 0.4})
        assert set(fig.data[0].labels) == {"AAA", "BBB"}


class TestBuildPortfolioCharts:
    def test_full_set_for_multi_asset(self, prices, stats):
        reg = ChartRegistry()
        records = charts.build_portfolio_charts(prices, {"AAA": 0.6, "BBB": 0.4}, stats, reg)
        kinds = [r.kind for r in records]
        assert kinds == ["price_history", "drawdown", "return_distribution", "correlation", "allocation"]
        assert len(reg) == 5
        for r in records:
            assert "data" in r.spec and "layout" in r.spec

    def test_single_asset_skips_correlation_and_allocation(self):
        idx = pd.date_range("2020-01-01", periods=120, freq="B")
        rng = np.random.default_rng(1)
        prices = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 120))}, index=idx)
        stats = quant.compute_portfolio_stats(prices, {"AAA": 1.0})
        reg = ChartRegistry()
        kinds = [r.kind for r in charts.build_portfolio_charts(prices, {"AAA": 1.0}, stats, reg)]
        assert "correlation" not in kinds and "allocation" not in kinds
        assert kinds == ["price_history", "drawdown", "return_distribution"]
