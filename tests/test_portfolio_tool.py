"""Tests for portfolio_tool.PortfolioAnalysisTool — offline, via an injected fake price
source (no yfinance, no network). Proves data -> quant -> MetricRegistry wiring and the
graceful handling of missing tickers / bad inputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import tavily_maxer as tm
from portfolio_tool import build_portfolio_tool

METRIC_RE = re.compile(r"\[metric:(\d+)\]")


class FakePriceSource:
    """Deterministic price history for known tickers; omits unknown ones."""

    name = "fake"

    def __init__(self, known, rows=260, seed=0):
        self.known = set(known)
        self.rows = rows
        self.seed = seed

    def fetch(self, tickers, period, interval):
        idx = pd.date_range("2020-01-01", periods=self.rows, freq="B")
        rng = np.random.default_rng(self.seed)
        data = {
            t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, self.rows))
            for t in tickers if t in self.known
        }
        return pd.DataFrame(data, index=idx)


def _tool(known=("AAA", "BBB", "SPY")):
    return build_portfolio_tool(FakePriceSource(known))


class TestAnalyzePortfolio:
    def test_registers_metrics_and_returns_labeled_text(self):
        tool = _tool()
        out = tool._run(["AAA", "BBB"], weights=[0.6, 0.4], var_confidence=0.99)
        ids_in_text = {int(m) for m in METRIC_RE.findall(out)}
        assert ids_in_text == tool.registry.ids()
        assert len(tool.registry) >= 10
        # every registered metric value is finite
        assert all(np.isfinite(m.value) for m in tool.registry.all_metrics())

    def test_equal_weight_when_weights_omitted(self):
        tool = _tool()
        out = tool._run(["AAA", "BBB"])
        assert "AAA 50.0%" in out and "BBB 50.0%" in out

    def test_missing_ticker_is_excluded_not_fatal(self):
        tool = _tool()
        out = tool._run(["AAA", "ZZZ"])
        assert "ZZZ" in out  # reported as excluded
        assert len(tool.registry) >= 10  # still computed on AAA

    def test_all_missing_returns_error(self):
        tool = _tool()
        out = tool._run(["ZZZ", "QQQ"])
        assert out.startswith("ERROR")
        assert len(tool.registry) == 0

    def test_weight_length_mismatch_returns_error(self):
        tool = _tool()
        out = tool._run(["AAA", "BBB"], weights=[1.0])
        assert out.startswith("ERROR")

    def test_benchmark_produces_beta_metric(self):
        tool = _tool()
        tool._run(["AAA", "BBB"], benchmark_ticker="SPY")
        assert any(m.key == "beta" for m in tool.registry.all_metrics())

    def test_no_benchmark_skips_beta(self):
        tool = _tool()
        tool._run(["AAA", "BBB"])
        assert not any(m.key == "beta" for m in tool.registry.all_metrics())

    def test_charts_are_built_and_referenceable(self):
        tool = _tool()
        out = tool._run(["AAA", "BBB"])
        chart_ids = {int(m) for m in re.findall(r"\[chart:(\d+)\]", out)}
        assert chart_ids == tool.chart_registry.ids()
        assert len(tool.chart_registry) == 5  # multi-asset full set
        for c in tool.chart_registry.all_charts():
            assert "data" in c.spec and "layout" in c.spec

    def test_include_charts_false_skips_chart_build(self):
        tool = _tool()
        out = tool._run(["AAA", "BBB"], include_charts=False)
        assert "[chart:" not in out
        assert len(tool.chart_registry) == 0
        assert len(tool.registry) >= 10  # metrics still computed


class TestToolValidationWiring:
    """End-to-end: metrics the tool registers validate; fabricated ones do not."""

    def test_referenced_metrics_validate_against_registry(self):
        tool = _tool()
        tool._run(["AAA", "BBB"])
        valid_id = min(tool.registry.ids())
        answer = tm.ResearchAnswer(
            answer=f"The Sharpe ratio is strong [metric:{valid_id}].",
            referenced_metric_ids=[valid_id],
        )
        result = tm.validate_artifacts(answer, tm.SourceRegistry(), tool.registry)
        assert result.valid

    def test_fabricated_metric_id_fails(self):
        tool = _tool()
        tool._run(["AAA", "BBB"])
        bad = max(tool.registry.ids()) + 999
        answer = tm.ResearchAnswer(answer=f"Made-up number [metric:{bad}].")
        result = tm.validate_artifacts(answer, tm.SourceRegistry(), tool.registry)
        assert not result.valid
        assert bad in result.missing_metric_ids

    def test_metric_reference_without_registry_is_invalid(self):
        # A research-only run (no metric registry) must reject any [metric:k].
        answer = tm.ResearchAnswer(answer="Sneaky [metric:1].")
        result = tm.validate_citations(answer, tm.SourceRegistry())
        assert not result.valid
        assert 1 in result.missing_metric_ids
