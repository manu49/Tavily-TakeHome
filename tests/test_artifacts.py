"""Tests for artifacts.MetricRegistry — the code-owned, citable metric ids."""

from __future__ import annotations

import math

import pytest

import quant as q
from artifacts import MetricRegistry


def _stats(**overrides):
    base = dict(
        weights={"A": 0.5, "B": 0.5},
        annualized_return=0.10,
        annualized_volatility=0.20,
        sharpe_ratio=0.50,
        sortino_ratio=0.70,
        max_drawdown=-0.30,
        var={"historical": 0.03, "parametric": 0.025, "monte_carlo": 0.031},
        cvar={"historical": 0.04, "parametric": 0.035},
        beta=None,
        inputs={"var_confidence": 0.99, "var_horizon": 1, "risk_free_rate": 0.04},
    )
    base.update(overrides)
    return q.PortfolioStats(**base)


class TestMetricRegistry:
    def test_register_assigns_sequential_ids(self):
        reg = MetricRegistry()
        a = reg.register("x", "X", 0.1, "ratio", "def")
        b = reg.register("y", "Y", 0.2, "ratio", "def")
        assert (a.id, b.id) == (1, 2)
        assert reg.ids() == {1, 2}

    def test_formatting_by_unit(self):
        reg = MetricRegistry()
        pct = reg.register("r", "Return", 0.1234, "percent", "d")
        ratio = reg.register("s", "Sharpe", 0.51, "ratio", "d")
        assert pct.formatted() == "12.34%"
        assert ratio.formatted() == "0.51"

    def test_register_portfolio_stats_creates_expected_metrics(self):
        reg = MetricRegistry()
        records = reg.register_portfolio_stats(_stats())
        keys = {r.key for r in records}
        # 5 scalars (beta skipped: None) + 3 VaR + 2 CVaR = 10
        assert len(records) == 10
        assert "beta" not in keys
        assert {"var_historical", "var_parametric", "var_monte_carlo"} <= keys
        assert {"cvar_historical", "cvar_parametric"} <= keys

    def test_nan_and_none_values_are_skipped(self):
        reg = MetricRegistry()
        records = reg.register_portfolio_stats(_stats(sortino_ratio=float("nan")))
        assert "sortino_ratio" not in {r.key for r in records}

    def test_beta_included_when_present(self):
        reg = MetricRegistry()
        records = reg.register_portfolio_stats(_stats(beta=1.15))
        beta = [r for r in records if r.key == "beta"]
        assert len(beta) == 1 and beta[0].value == pytest.approx(1.15)

    def test_var_name_includes_confidence(self):
        reg = MetricRegistry()
        reg.register_portfolio_stats(_stats())
        var_hist = next(m for m in reg.all_metrics() if m.key == "var_historical")
        assert "99%" in var_hist.name

    def test_to_markdown_lists_metric_ids(self):
        reg = MetricRegistry()
        reg.register_portfolio_stats(_stats())
        md = reg.to_markdown()
        assert "[metric:1]" in md
        assert md.count("[metric:") == len(reg)

    def test_empty_registry_markdown(self):
        assert MetricRegistry().to_markdown() == "_No metrics computed._"

    def test_get_and_len(self):
        reg = MetricRegistry()
        r = reg.register("x", "X", 1.0, "ratio", "d")
        assert reg.get(r.id) is r
        assert reg.get(999) is None
        assert len(reg) == 1
