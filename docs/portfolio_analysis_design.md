# Design — tavily_maxer for Multi-Asset Portfolio Managers

## Motivation

tavily_maxer today answers research questions and proves every claim against a real
retrieved source. Portfolio managers (PMs) need more than prose: for a question like
*"What is the predicted return of the S&P 500 over the next 12 months based on the last 5
years?"* or *"Analyze the risk in my portfolio,"* the deliverable is **quantitative metrics
and charts**, computed correctly, sitting next to grounded narrative.

The credible pitch is the same one that already differentiates tavily_maxer — *nothing is
improvised* — extended from citations to **numbers and figures**: every Sharpe ratio, every
VaR, every plot traces to a deterministic computation over real data, and the model can
reference those artifacts but can never fabricate them.

## Design principle (the throughline)

> The LLM orchestrates and explains. **Code computes.** The model decides *what* to analyze
> and narrates *what it means*; it never produces a metric value, a data point, or a
> forecast number itself.

This is the existing `SourceRegistry` pattern — citation ids are assigned in code so the
model can't hallucinate a URL — generalized to two new artifact classes: **metrics** and
**charts**. Validation already enforces "every cited id is real"; we extend it to "every
referenced metric/chart id is real and code-produced."

### Note on "fine-tuning"

This needs almost no model weight fine-tuning. The capability comes from tool-augmentation +
deterministic compute + structured artifacts + a specialized system prompt. Weight
fine-tuning would only later help house tone/report format, and it cannot fix quantitative
correctness — so it is explicitly out of scope for v1.

## Architecture

```
question (+ optional uploaded portfolio file)
   │
   ▼
create_agent (ReAct loop)  ── system prompt specialized for portfolio analysis
   │
   │  tools (all code-owned, deterministic where it counts):
   │    • tavily_search            → qualitative context, analyst estimates (cited)
   │    • get_price_history        → adjusted-close time series for tickers
   │    • analyze_portfolio        → returns, vol, Sharpe/Sortino, VaR/CVaR, drawdown,
   │                                  beta, correlations  ── MetricRegistry
   │    • make_chart               → matplotlib figures           ── ChartRegistry
   │    • get_uploaded_portfolio   → the parsed, validated holdings
   │
   ▼ (response_format = AnalysisAnswer)
structured answer: { answer_markdown,
                     cited_source_ids[], referenced_metric_ids[], referenced_chart_ids[] }
   │   answer markdown embeds inline [n], [metric:k], [chart:j] placeholders
   ▼
validate_artifacts(answer, sources, metrics, charts)
   │   every referenced id exists in its registry → pass (no hallucinated number/plot)
   ▼
render (web: markdown + inline charts + metrics table + Sources + validation badge)
   ▼
trace (LangSmith + JSONL: tools called, metrics computed, validation, latency, tokens)
```

## Components

### 1. Market data — `marketdata.py`
`get_price_history(tickers, period, interval)` returns a tidy adjusted-close frame.

- Backed by `yfinance` for the demo; abstracted behind a `PriceSource` protocol so a
  licensed feed (Polygon, FMP, Bloomberg) drops in without touching the agent.
- Caches by `(ticker, period, interval)` for the run; records the data vintage (as-of date,
  source) so every downstream metric is reproducible and auditable.
- Returns a structured error the model can act on (e.g. unknown ticker) rather than raising.

### 2. Quant engine — `quant.py` (the deterministic core, no LLM)
Pure functions over a returns matrix. **This is where correctness lives and where the test
suite is densest.**

- Returns: simple vs log (explicit), adjusted-close based.
- Risk/return: annualized return (CAGR), annualized volatility, **Sharpe**, **Sortino**,
  max drawdown, **beta** vs a benchmark, rolling metrics.
- Tail risk: **VaR** by three methods — historical (empirical percentile), parametric
  (μ − z·σ), Monte-Carlo (simulated paths) — plus **CVaR / Expected Shortfall**. Confidence
  (95/99%) and horizon (with √-time scaling) are explicit parameters.
- Portfolio level: weights from quantities × prices, covariance matrix, portfolio variance,
  correlation matrix, marginal/component VaR, optional efficient frontier.
- Every result carries its **definition string and the exact inputs used** (risk-free rate,
  confidence, horizon, return basis, data vintage) so a metric is self-documenting.

### 3. MetricRegistry — `artifacts.py`
Mirrors `SourceRegistry`. The `analyze_portfolio` tool runs `quant.py` and registers each
result: `{id, name, value, unit, definition, inputs}`. The model receives them pre-labeled
(`[metric:3] 1-day 99% Historical VaR = -2.4% (...)`) and references them by id. The number
the user sees is the registered number — the model has no path to alter it.

### 4. ChartRegistry + `make_chart` — `charts.py`
`make_chart(kind, args)` renders a matplotlib figure (Agg backend) and registers it
`{id, kind, title, caption, png_b64}`. Supported kinds: price history, drawdown curve,
return distribution with VaR/CVaR markers, correlation heatmap, allocation pie, efficient
frontier. Charts are built **only from registered metrics / fetched series**, never from
model-supplied numbers. The model embeds `[chart:2]`; the frontend swaps in the image.

### 5. Structured schema — extend `ResearchAnswer` → `AnalysisAnswer`
```python
class AnalysisAnswer(BaseModel):
    answer: str                       # markdown w/ inline [n], [metric:k], [chart:j]
    cited_source_ids: list[int] = []
    referenced_metric_ids: list[int] = []
    referenced_chart_ids: list[int] = []
```
Backward compatible: a pure web-research question simply leaves the new lists empty.

### 6. Validation — extend `validate_citations` → `validate_artifacts`
Deterministic, post-hoc, runs every time:
- Every `[n]` resolves in `SourceRegistry` (unchanged).
- Every `[metric:k]` resolves in `MetricRegistry`; every `[chart:j]` in `ChartRegistry`.
- No stray/improvised markers (the existing `【…】` guard, generalized).
- Soft check: high-salience computed metrics that were never referenced are flagged (the
  model ignored a result it asked for) — surfaced as a warning, not a hard fail.

### 7. Portfolio upload — `portfolio.py` + new endpoint
- `POST /api/upload` accepts `.xlsx` / `.csv` / `.txt`; `pandas` / `openpyxl` parse it into a
  validated `Portfolio` (ticker, quantity or weight, optional cost basis, currency).
- Flexible column mapping + validation report (unknown tickers, weights ≠ 100%, missing
  prices) returned to the UI before any analysis runs.
- The `Portfolio` is passed into the run and exposed via `get_uploaded_portfolio()`.

## Quantitative honesty (trust, not just correctness)

- **Forecasts have no single true value.** For "predicted 12-month return," the agent
  presents (a) historical CAGR + the return *distribution* + scenario ranges (computed) and
  (b) sourced analyst estimates via Tavily (cited) — explicitly labeled as assumptions /
  estimates. It must not assert a fabricated point forecast. Same ethos as citations.
- **Surface the assumptions.** Risk-free rate, VaR confidence/horizon, return basis, and data
  vintage are shown with every figure. Defaults are sane and explicit, chosen by tool
  parameters, never silently by the model.
- **Not investment advice.** A standing disclaimer; the tool analyzes, it does not recommend
  trades.

## Observability & evaluation

- Extend the JSONL/LangSmith record with: tickers fetched, data vintage, metrics computed
  (id→value), charts produced, artifact-validation result.
- Evals (`evals/`) gain a **numerical-correctness** tier: golden portfolios with
  known-answer metrics (Sharpe, VaR, drawdown) computed independently (e.g. a spreadsheet /
  reference implementation), asserted to a tolerance. This is a deterministic gate
  independent of any LLM — the most important new test surface.

## Deployment implications (important)

`numpy + pandas + scipy + matplotlib + yfinance + openpyxl` is heavy and stateful:

- Likely **exceeds Vercel's serverless limits** (250 MB unzipped) and worsens cold starts;
  matplotlib + uploads also want a writable filesystem and session state, which serverless
  makes awkward.
- **Recommendation:** move this tier to a **container host** (Render / Railway / Fly /
  Cloud Run) with a persistent process and a small session store, or split compute out from
  a thin Vercel front end. The current single-file demo stays a good *interface* prototype.

## File structure (proposed)

```
tavily_maxer.py        # agent factory, run loop, validation (extended)
marketdata.py          # PriceSource protocol + yfinance impl + get_price_history tool
quant.py               # deterministic metrics: returns, Sharpe, VaR/CVaR, drawdown, beta...
charts.py              # ChartRegistry + make_chart (matplotlib)
artifacts.py           # MetricRegistry (+ shared base with SourceRegistry)
portfolio.py           # upload parsing + Portfolio model + validation
webapp.py              # + /api/upload, chart/metric rendering in the page
evals/                 # + numerical-correctness golden cases
tests/                 # + dense quant.py unit tests (known-answer)
```

## Build phases

| Phase | Output |
|---|---|
| 1. Quant core | `quant.py` + `marketdata.py` + `MetricRegistry`, dense known-answer tests. No LLM/UI. |
| 2. Agent wiring | `analyze_portfolio` tool, `AnalysisAnswer` schema, `validate_artifacts`. |
| 3. Charts | `charts.py` + `make_chart`, inline chart rendering in the page. |
| 4. Upload | `/api/upload`, `portfolio.py`, dropzone + validation report in UI. |
| 5. Honesty + evals | forecast framing, assumptions surfacing, numerical-correctness eval tier. |
| 6. Deploy | container host migration; session state; observability fields. |

## Open questions / decisions to confirm

1. **Data source:** yfinance for the demo, or wire a licensed feed now (affects rate limits,
   accuracy, terms)?
2. **Asset classes:** equities/ETFs first, or also FX, fixed income, crypto (each needs its
   own data + risk conventions)?
3. **VaR defaults:** confidence (95% vs 99%), horizon (1-day vs 10-day), method emphasis.
4. **Benchmark:** default to S&P 500 for beta, or PM-specified per portfolio?
5. **Statefulness:** how long should an uploaded portfolio persist (single request, a
   session, or saved)? Drives the hosting/session design.
```
