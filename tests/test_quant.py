"""Known-answer tests for quant.py.

Every metric is checked against a hand-computed value on a tiny, fully-specified input, so
a regression in a formula fails loudly. This is the deterministic correctness gate the
design doc calls the most important new test surface — no network, no LLM.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import quant as q

APPROX = dict(rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------------------

class TestReturns:
    def test_simple_returns(self):
        prices = pd.Series([100.0, 110.0, 99.0])
        result = q.simple_returns(prices).tolist()
        assert result == pytest.approx([0.1, -0.1])

    def test_log_returns(self):
        prices = pd.Series([100.0, 110.0])
        assert q.log_returns(prices).iloc[0] == pytest.approx(math.log(1.1))

    def test_simple_returns_drops_first_row(self):
        assert len(q.simple_returns(pd.Series([1.0, 2.0, 3.0]))) == 2


# --------------------------------------------------------------------------------------
# Risk / return
# --------------------------------------------------------------------------------------

class TestRiskReturn:
    def test_annualized_return_geometric(self):
        # 1.1 * 0.9 = 0.99 over 2 periods, ppy=2 -> 0.99^1 - 1
        assert q.annualized_return([0.1, -0.1], periods_per_year=2) == pytest.approx(-0.01)

    def test_annualized_return_constant(self):
        assert q.annualized_return([0.05] * 4, periods_per_year=4) == pytest.approx(1.05**4 - 1)

    def test_annualized_volatility_scales_with_sqrt_periods(self):
        rets = [0.1, 0.2, 0.3]  # sample std = 0.1
        assert q.annualized_volatility(rets, 1) == pytest.approx(0.1)
        assert q.annualized_volatility(rets, 4) == pytest.approx(0.2)

    def test_sharpe_ratio(self):
        assert q.sharpe_ratio([0.1, 0.2, 0.3], 0.0, 1) == pytest.approx(2.0)
        assert q.sharpe_ratio([0.1, 0.2, 0.3], 0.0, 4) == pytest.approx(4.0)

    def test_sharpe_ratio_with_risk_free(self):
        # rf annual 0.4, ppy 4 -> rf_period 0.1; excess mean 0.1, std 0.1, *sqrt(4)
        assert q.sharpe_ratio([0.1, 0.2, 0.3], 0.4, 4) == pytest.approx(2.0)

    def test_sortino_only_penalizes_downside(self):
        # excess [0.2,-0.1]; downside_dev = sqrt(mean([0, 0.01])) = sqrt(0.005)
        expected = 0.05 / math.sqrt(0.005)
        assert q.sortino_ratio([0.2, -0.1], 0.0, 1) == pytest.approx(expected)

    def test_zero_volatility_is_nan_not_crash(self):
        assert math.isnan(q.sharpe_ratio([0.05, 0.05, 0.05]))

    def test_max_drawdown(self):
        prices = pd.Series([100.0, 120.0, 90.0, 110.0])  # trough 90 vs peak 120
        assert q.max_drawdown(prices) == pytest.approx(-0.25)

    def test_drawdown_series_nonpositive(self):
        dd = q.drawdown_series(pd.Series([100.0, 120.0, 90.0, 110.0]))
        assert (dd <= 1e-12).all()
        assert dd.iloc[0] == pytest.approx(0.0)

    def test_beta_of_2x_benchmark_is_2(self):
        bench = [0.01, -0.02, 0.03]
        asset = [2 * x for x in bench]
        assert q.beta(asset, bench) == pytest.approx(2.0)

    def test_beta_of_benchmark_with_itself_is_1(self):
        bench = [0.01, -0.02, 0.03, 0.005]
        assert q.beta(bench, bench) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# VaR / CVaR (positive loss fractions)
# --------------------------------------------------------------------------------------

TAIL = [-0.05, -0.03, -0.01, 0.0, 0.02, 0.04]


class TestVaR:
    def test_historical_var(self):
        # np.quantile(TAIL, 0.05) = -0.045 -> VaR 0.045
        assert q.historical_var(TAIL, 0.95) == pytest.approx(0.045)

    def test_historical_var_horizon_scaling(self):
        assert q.historical_var(TAIL, 0.95, horizon=4) == pytest.approx(0.09)  # *sqrt(4)

    def test_parametric_var(self):
        # mean 0, std 0.01, z(0.975)=1.959964 -> 0.0196
        assert q.parametric_var([-0.01, 0.0, 0.01], 0.975) == pytest.approx(0.0195996, abs=1e-6)

    def test_var_never_negative(self):
        # an all-positive return series should clamp to 0, not a negative "loss"
        assert q.historical_var([0.01, 0.02, 0.03], 0.95) == 0.0

    def test_monte_carlo_is_deterministic_with_seed(self):
        a = q.monte_carlo_var(TAIL, 0.95, seed=7)
        b = q.monte_carlo_var(TAIL, 0.95, seed=7)
        assert a == b and a >= 0.0

    def test_higher_confidence_means_larger_var(self):
        assert q.parametric_var(TAIL, 0.99) > q.parametric_var(TAIL, 0.95)

    def test_dispatch_matches_direct_calls(self):
        assert q.value_at_risk(TAIL, 0.95, method="historical") == q.historical_var(TAIL, 0.95)
        assert q.value_at_risk(TAIL, 0.95, method="parametric") == q.parametric_var(TAIL, 0.95)

    def test_dispatch_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            q.value_at_risk(TAIL, method="psychic")


class TestCVaR:
    def test_historical_cvar(self):
        # tail at/below -0.045 is just {-0.05}; mean -0.05 -> CVaR 0.05
        assert q.historical_cvar(TAIL, 0.95) == pytest.approx(0.05)

    def test_cvar_at_least_var(self):
        assert q.historical_cvar(TAIL, 0.95) >= q.historical_var(TAIL, 0.95)

    def test_parametric_cvar(self):
        assert q.parametric_cvar([-0.01, 0.0, 0.01], 0.975) == pytest.approx(0.023378, abs=1e-5)


# --------------------------------------------------------------------------------------
# Portfolio aggregation
# --------------------------------------------------------------------------------------

class TestPortfolio:
    def test_weights_from_holdings(self):
        w = q.weights_from_holdings({"A": 10, "B": 5}, {"A": 100, "B": 200})
        assert w == pytest.approx({"A": 0.5, "B": 0.5})

    def test_weights_raise_on_zero_value(self):
        with pytest.raises(ValueError):
            q.weights_from_holdings({"A": 0}, {"A": 100})

    def test_portfolio_returns_weighted_sum(self):
        rets = pd.DataFrame({"A": [0.1, -0.1], "B": [0.2, 0.0]})
        port = q.portfolio_returns(rets, {"A": 0.5, "B": 0.5})
        assert port.tolist() == pytest.approx([0.15, -0.05])

    def test_correlation_matrix_extremes(self):
        a = pd.Series([0.01, 0.02, 0.03])
        rets = pd.DataFrame({"A": a, "B": 2 * a, "C": -a})
        corr = q.correlation_matrix(rets)
        assert corr.loc["A", "B"] == pytest.approx(1.0)
        assert corr.loc["A", "C"] == pytest.approx(-1.0)

    def test_portfolio_volatility_diagonal_cov(self):
        cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
        assert q.portfolio_volatility({"A": 1.0, "B": 0.0}, cov) == pytest.approx(0.2)
        assert q.portfolio_volatility({"A": 0.5, "B": 0.5}, cov) == pytest.approx(math.sqrt(0.0325))


# --------------------------------------------------------------------------------------
# Bundled stats
# --------------------------------------------------------------------------------------

class TestComputePortfolioStats:
    def _prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        a = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
        b = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, len(idx)))
        return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)

    def test_returns_full_metric_set(self):
        stats = q.compute_portfolio_stats(self._prices(), {"AAA": 0.6, "BBB": 0.4})
        d = stats.to_dict()
        assert set(d["var"]) == {"historical", "parametric", "monte_carlo"}
        assert set(d["cvar"]) == {"historical", "parametric"}
        for key in ("annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"):
            assert math.isfinite(d[key])
        assert d["beta"] is None  # no benchmark supplied
        assert d["inputs"]["n_observations"] == 299

    def test_beta_computed_when_benchmark_supplied(self):
        prices = self._prices()
        bench_returns = q.simple_returns(prices["AAA"])  # portfolio correlated with benchmark
        stats = q.compute_portfolio_stats(
            prices, {"AAA": 1.0, "BBB": 0.0}, benchmark_returns=bench_returns
        )
        # Portfolio is 100% AAA and benchmark IS AAA's returns -> beta ~ 1.
        assert stats.beta == pytest.approx(1.0, abs=1e-6)

    def test_inputs_record_assumptions(self):
        stats = q.compute_portfolio_stats(
            self._prices(), {"AAA": 0.5, "BBB": 0.5}, risk_free_rate=0.04, var_confidence=0.99
        )
        assert stats.inputs["risk_free_rate"] == 0.04
        assert stats.inputs["var_confidence"] == 0.99
