"""
quant.py — the deterministic quantitative engine for portfolio analysis.

This is the *code-owned* core from the design doc: every number a portfolio manager sees
(Sharpe, VaR, drawdown, beta, ...) is computed here, by formula, over real price data. The
LLM never produces these values; it only decides what to compute and explains the result.
Keeping this module pure (NumPy/pandas/SciPy in, plain floats/dataclasses out — no network,
no LLM, no global state) is what makes it exhaustively unit-testable against known answers.

Conventions (stated once, applied everywhere):
  * Returns are simple (arithmetic) unless a function says "log".
  * `periods_per_year` annualizes; default 252 (daily trading days). Use 52 weekly, 12
    monthly.
  * Volatility uses the sample standard deviation (ddof=1).
  * VaR and CVaR/Expected Shortfall are returned as **positive loss fractions**: a 1-day 95%
    VaR of 0.024 means "a loss of 2.4% of value is not expected to be exceeded with 95%
    confidence over 1 day." Horizon scaling uses the square-root-of-time rule.
  * Max drawdown is returned as a **negative** fraction (e.g. -0.25 for a 25% peak-to-trough
    decline).
  * Every public function documents the exact formula it implements so a metric is
    self-explaining when surfaced to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
PERIODS_PER_YEAR: Dict[str, int] = {"1d": 252, "1wk": 52, "1mo": 12}

# Dispersion below this is treated as "no variation" so ratios don't explode on a
# constant series (whose sample std is a floating-point ~1e-17 rather than exactly 0).
_ZERO_DISPERSION = 1e-12

VaRMethod = Literal["historical", "parametric", "monte_carlo"]


# --------------------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------------------

def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Period-over-period simple returns: r_t = P_t / P_{t-1} - 1. Drops the first NaN row."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Continuously-compounded returns: r_t = ln(P_t / P_{t-1}). Drops the first NaN row."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def _as_1d_array(returns: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    return arr[~np.isnan(arr)]


# --------------------------------------------------------------------------------------
# Risk / return metrics
# --------------------------------------------------------------------------------------

def annualized_return(
    returns: Sequence[float] | pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Geometric annualized return (CAGR-equivalent) from a series of periodic returns:

        (∏ (1 + r_t))^(periods_per_year / n) - 1
    """
    r = _as_1d_array(returns)
    if r.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    return growth ** (periods_per_year / r.size) - 1.0


def annualized_volatility(
    returns: Sequence[float] | pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized standard deviation of returns: std(r, ddof=1) * sqrt(periods_per_year)."""
    r = _as_1d_array(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio.

        sharpe = mean(r - rf_period) / std(r, ddof=1) * sqrt(periods_per_year)

    `risk_free_rate` is an *annual* rate; it is converted to a per-period rate by simple
    division (rf_period = risk_free_rate / periods_per_year).
    """
    r = _as_1d_array(returns)
    if r.size < 2:
        return float("nan")
    rf_period = risk_free_rate / periods_per_year
    sd = np.std(r, ddof=1)
    if sd < _ZERO_DISPERSION:
        return float("nan")
    return float((np.mean(r - rf_period) / sd) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sortino ratio: like Sharpe but penalizing only downside deviation.

        sortino = mean(r - rf_period) / downside_dev * sqrt(periods_per_year)

    where downside_dev = sqrt(mean(min(r - rf_period, 0)^2)) (returns below the target are
    the only ones that contribute to risk).
    """
    r = _as_1d_array(returns)
    if r.size < 2:
        return float("nan")
    rf_period = risk_free_rate / periods_per_year
    excess = r - rf_period
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev < _ZERO_DISPERSION:
        return float("nan")
    return float((np.mean(excess) / downside_dev) * np.sqrt(periods_per_year))


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Drawdown at each point: P_t / (running max of P up to t) - 1 (<= 0 everywhere)."""
    running_max = prices.cummax()
    return prices / running_max - 1.0


def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough decline as a negative fraction (e.g. -0.25 = -25%)."""
    if len(prices) == 0:
        return float("nan")
    return float(drawdown_series(prices).min())


def beta(
    asset_returns: Sequence[float] | pd.Series,
    benchmark_returns: Sequence[float] | pd.Series,
) -> float:
    """CAPM beta: cov(asset, benchmark) / var(benchmark), using sample (ddof=1) moments.

    The two series are aligned by truncating to their common length from the start; pass
    already-aligned, equal-length returns for exact results.
    """
    a = _as_1d_array(asset_returns)
    b = _as_1d_array(benchmark_returns)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    var_b = np.var(b, ddof=1)
    if var_b < _ZERO_DISPERSION:
        return float("nan")
    cov = np.cov(a, b, ddof=1)[0, 1]
    return float(cov / var_b)


# --------------------------------------------------------------------------------------
# Value at Risk and Expected Shortfall (positive loss fractions)
# --------------------------------------------------------------------------------------

def historical_var(
    returns: Sequence[float] | pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Historical (empirical) VaR: the negative of the (1 - confidence) return quantile,
    scaled to the horizon by sqrt(horizon).

        VaR = -quantile(returns, 1 - confidence) * sqrt(horizon)

    Clamped at 0 (a non-negative loss). Uses linear-interpolation quantiles (NumPy default).
    """
    r = _as_1d_array(returns)
    if r.size == 0:
        return float("nan")
    q = float(np.quantile(r, 1.0 - confidence))
    return max(0.0, -q * np.sqrt(horizon))


def parametric_var(
    returns: Sequence[float] | pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Variance-covariance (Gaussian) VaR.

    Assuming r ~ Normal(μ, σ), the (1 - confidence) lower quantile is μ - z·σ with
    z = Φ⁻¹(confidence). As a positive loss, scaled to the horizon:

        VaR = z·σ·sqrt(horizon) - μ·horizon
    """
    r = _as_1d_array(returns)
    if r.size < 2:
        return float("nan")
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    z = norm.ppf(confidence)
    return max(0.0, z * sigma * np.sqrt(horizon) - mu * horizon)


def monte_carlo_var(
    returns: Sequence[float] | pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> float:
    """Monte-Carlo VaR by bootstrapping the empirical return distribution.

    Draws `n_sims` horizon-length paths by sampling observed returns with replacement,
    compounds each path to a horizon return, and takes the negative (1 - confidence)
    quantile of the simulated horizon-return distribution. Deterministic for a fixed `seed`.
    """
    r = _as_1d_array(returns)
    if r.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_sims, horizon), replace=True)
    horizon_returns = np.prod(1.0 + draws, axis=1) - 1.0
    q = float(np.quantile(horizon_returns, 1.0 - confidence))
    return max(0.0, -q)


def value_at_risk(
    returns: Sequence[float] | pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    method: VaRMethod = "historical",
    **kwargs,
) -> float:
    """Dispatch to the requested VaR method. Returns a positive loss fraction."""
    if method == "historical":
        return historical_var(returns, confidence, horizon)
    if method == "parametric":
        return parametric_var(returns, confidence, horizon)
    if method == "monte_carlo":
        return monte_carlo_var(returns, confidence, horizon, **kwargs)
    raise ValueError(f"Unknown VaR method: {method!r}")


def historical_cvar(
    returns: Sequence[float] | pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Historical Expected Shortfall: the mean of returns at or below the (1 - confidence)
    quantile (the average loss in the tail), as a positive loss scaled by sqrt(horizon)."""
    r = _as_1d_array(returns)
    if r.size == 0:
        return float("nan")
    q = np.quantile(r, 1.0 - confidence)
    tail = r[r <= q]
    if tail.size == 0:
        return max(0.0, -float(q) * np.sqrt(horizon))
    return max(0.0, -float(np.mean(tail)) * np.sqrt(horizon))


def parametric_cvar(
    returns: Sequence[float] | pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Gaussian Expected Shortfall.

        ES = z_pdf/(1 - confidence) · σ·sqrt(horizon) - μ·horizon

    where z_pdf = φ(Φ⁻¹(confidence)) is the standard-normal density at the VaR quantile.
    """
    r = _as_1d_array(returns)
    if r.size < 2:
        return float("nan")
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    alpha = 1.0 - confidence
    es_factor = norm.pdf(norm.ppf(confidence)) / alpha
    return max(0.0, es_factor * sigma * np.sqrt(horizon) - mu * horizon)


# --------------------------------------------------------------------------------------
# Portfolio aggregation
# --------------------------------------------------------------------------------------

def weights_from_holdings(
    quantities: Mapping[str, float], latest_prices: Mapping[str, float]
) -> Dict[str, float]:
    """Convert share quantities + latest prices into market-value weights summing to 1."""
    values = {t: quantities[t] * latest_prices[t] for t in quantities}
    total = sum(values.values())
    if total == 0:
        raise ValueError("Total portfolio market value is zero; cannot compute weights.")
    return {t: v / total for t, v in values.items()}


def portfolio_returns(returns: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Weighted portfolio return series (fixed weights, rebalanced each period).

        r_p,t = Σ_i w_i · r_i,t
    """
    w = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    return returns.mul(w, axis=1).sum(axis=1)


def covariance_matrix(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Annualized sample covariance matrix of the asset returns."""
    return returns.cov(ddof=1) * periods_per_year


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of the asset returns (annualization-invariant)."""
    return returns.corr()


def portfolio_volatility(
    weights: Mapping[str, float], cov_annualized: pd.DataFrame
) -> float:
    """Annualized portfolio volatility from weights and an annualized covariance matrix:

        σ_p = sqrt(wᵀ Σ w)
    """
    cols = list(cov_annualized.columns)
    w = pd.Series(weights, dtype=float).reindex(cols).fillna(0.0).to_numpy()
    sigma = cov_annualized.to_numpy()
    return float(np.sqrt(w @ sigma @ w))


# --------------------------------------------------------------------------------------
# Bundled result
# --------------------------------------------------------------------------------------

@dataclass
class PortfolioStats:
    """Everything the analyze_portfolio tool will register as metrics (Phase 2). Plain,
    JSON-friendly numbers plus the inputs used, so each value is reproducible/auditable."""

    weights: Dict[str, float]
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var: Dict[str, float]               # method -> positive loss fraction
    cvar: Dict[str, float]              # method -> positive loss fraction
    beta: float | None
    inputs: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "var": self.var,
            "cvar": self.cvar,
            "beta": self.beta,
            "inputs": self.inputs,
        }


def compute_portfolio_stats(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    var_confidence: float = 0.95,
    var_horizon: int = 1,
    benchmark_returns: pd.Series | None = None,
    mc_seed: int | None = 42,
) -> PortfolioStats:
    """Compute the full metric set for a weighted portfolio from a price panel.

    `prices` is a DataFrame of adjusted closes (one column per ticker). Weights are matched
    to columns by name. All three VaR/CVaR methods are reported so the PM can compare them.
    """
    rets = simple_returns(prices)
    if isinstance(rets, pd.Series):
        rets = rets.to_frame()

    port_rets = portfolio_returns(rets, weights)
    # Reconstruct a portfolio wealth index (starts at 1.0) for the drawdown calc.
    wealth = (1.0 + port_rets).cumprod()

    b = None
    if benchmark_returns is not None:
        aligned = pd.concat([port_rets, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) >= 2:
            b = beta(aligned.iloc[:, 0], aligned.iloc[:, 1])

    return PortfolioStats(
        weights=dict(weights),
        annualized_return=annualized_return(port_rets, periods_per_year),
        annualized_volatility=annualized_volatility(port_rets, periods_per_year),
        sharpe_ratio=sharpe_ratio(port_rets, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(port_rets, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(wealth),
        var={
            "historical": historical_var(port_rets, var_confidence, var_horizon),
            "parametric": parametric_var(port_rets, var_confidence, var_horizon),
            "monte_carlo": monte_carlo_var(port_rets, var_confidence, var_horizon, seed=mc_seed),
        },
        cvar={
            "historical": historical_cvar(port_rets, var_confidence, var_horizon),
            "parametric": parametric_cvar(port_rets, var_confidence, var_horizon),
        },
        beta=b,
        inputs={
            "risk_free_rate": risk_free_rate,
            "periods_per_year": periods_per_year,
            "var_confidence": var_confidence,
            "var_horizon": var_horizon,
            "n_observations": int(len(port_rets)),
            "tickers": list(prices.columns),
        },
    )
