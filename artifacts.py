"""
artifacts.py — MetricRegistry: code-owned, stably-numbered quantitative metrics.

This is the numeric analogue of tavily_maxer's SourceRegistry. Where SourceRegistry hands
the model citation ids it can't fabricate, MetricRegistry hands the model *metric* ids
([metric:k]) for values that were actually computed by quant.py. The model may reference a
metric by id and explain it, but the number the user sees is the registered number — the
LLM has no path to invent or alter a Sharpe ratio or a VaR.

Deliberately lightweight (no numpy/pandas/LLM imports) so it lives in the core tier and is
trivially testable. It consumes a plain stats object (duck-typed: the PortfolioStats from
quant.py) and reads attributes off it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _is_number(x: Any) -> bool:
    try:
        return x is not None and x == x and abs(float(x)) != float("inf")  # x == x rejects NaN
    except (TypeError, ValueError):
        return False


@dataclass
class MetricRecord:
    id: int
    key: str          # machine key, e.g. "sharpe_ratio"
    name: str         # human label, e.g. "Sharpe ratio (annualized)"
    value: float
    unit: str         # "percent" | "ratio"
    definition: str
    inputs: Dict[str, Any] = field(default_factory=dict)

    def formatted(self) -> str:
        if self.unit == "percent":
            return f"{self.value * 100:.2f}%"
        if self.unit == "ratio":
            return f"{self.value:.2f}"
        return f"{self.value}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "name": self.name,
            "value": self.value,
            "formatted": self.formatted(),
            "unit": self.unit,
            "definition": self.definition,
            "inputs": self.inputs,
        }


class MetricRegistry:
    """Single source of truth for metrics computed during one agent run. Assigns each a
    stable, code-owned integer id the model references as [metric:id]."""

    def __init__(self) -> None:
        self._by_id: Dict[int, MetricRecord] = {}
        self._next_id = 1

    def register(
        self, key: str, name: str, value: float, unit: str, definition: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> MetricRecord:
        record = MetricRecord(
            id=self._next_id, key=key, name=name, value=float(value),
            unit=unit, definition=definition, inputs=inputs or {},
        )
        self._by_id[record.id] = record
        self._next_id += 1
        return record

    def register_portfolio_stats(self, stats: Any) -> List[MetricRecord]:
        """Turn a quant.PortfolioStats into individually-citable metric records.

        Non-finite values (e.g. a NaN Sharpe on a too-short series, or a None beta when no
        benchmark was given) are skipped so the model is never offered a meaningless id.
        """
        inp = dict(getattr(stats, "inputs", {}) or {})
        conf = inp.get("var_confidence", 0.95)
        horizon = inp.get("var_horizon", 1)
        conf_pct = f"{conf * 100:.0f}%"
        horizon_label = f"{horizon}-day" if horizon == 1 else f"{horizon}-day"

        added: List[MetricRecord] = []

        def add(key, name, value, unit, definition):
            if _is_number(value):
                added.append(self.register(key, name, value, unit, definition, inp))

        add("annualized_return", "Annualized return (CAGR)", getattr(stats, "annualized_return", None),
            "percent", "Geometric annualized return over the sample period.")
        add("annualized_volatility", "Annualized volatility", getattr(stats, "annualized_volatility", None),
            "percent", "Annualized standard deviation of returns.")
        add("sharpe_ratio", "Sharpe ratio (annualized)", getattr(stats, "sharpe_ratio", None),
            "ratio", f"Excess return per unit of total volatility, rf={inp.get('risk_free_rate', 0.0)}.")
        add("sortino_ratio", "Sortino ratio (annualized)", getattr(stats, "sortino_ratio", None),
            "ratio", "Excess return per unit of downside deviation.")
        add("max_drawdown", "Maximum drawdown", getattr(stats, "max_drawdown", None),
            "percent", "Worst peak-to-trough decline over the sample period.")
        add("beta", "Beta vs benchmark", getattr(stats, "beta", None),
            "ratio", "Sensitivity of portfolio returns to the benchmark (CAPM beta).")

        var = getattr(stats, "var", {}) or {}
        cvar = getattr(stats, "cvar", {}) or {}
        for method, value in var.items():
            add(f"var_{method}", f"VaR ({conf_pct}, {horizon_label}, {method})", value, "percent",
                f"Value at Risk ({method}): loss not expected to be exceeded with {conf_pct} "
                f"confidence over {horizon_label}.")
        for method, value in cvar.items():
            add(f"cvar_{method}", f"CVaR / Expected Shortfall ({conf_pct}, {horizon_label}, {method})",
                value, "percent",
                f"Conditional VaR ({method}): average loss in the worst {100 - conf * 100:.0f}% of cases.")

        return added

    def get(self, metric_id: int) -> Optional[MetricRecord]:
        return self._by_id.get(metric_id)

    def ids(self) -> set[int]:
        return set(self._by_id.keys())

    def all_metrics(self) -> List[MetricRecord]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def to_markdown(self) -> str:
        """Pre-labeled listing the model can cite from (mirrors SourceRegistry.to_markdown)."""
        if not self._by_id:
            return "_No metrics computed._"
        return "\n".join(
            f"[metric:{m.id}] {m.name} = {m.formatted()} — {m.definition}"
            for m in self.all_metrics()
        )

    def __len__(self) -> int:
        return len(self._by_id)
