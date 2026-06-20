# Tavily Maxer — Intelligence for Portfolio Managers

**Live app: https://tavily-take-home.vercel.app/**

The ultimate intelligent portfolio research and management tool, powered by **Tavily Search** —
real-time web retrieval purpose-built for AI agents that returns clean, relevant,
citation-ready content (not raw HTML), with fine control over search depth, recency,
domains, and a finance-tuned topic. That grounding is what lets every answer here trace back
to a real, retrieved source.

## Productionized Webapp

![The Tavily Maxer landing page on Vercel — "Intelligence for Portfolio Managers" with the Research / Portfolio mode toggle](Demo/landing_page.png)

<video src="https://github.com/manu49/Tavily-TakeHome/raw/main/Demo/v3_Tavily_Maxer.mp4" controls width="100%"></video>

> If the player doesn't load inline, [watch/download the demo video](Demo/v3_Tavily_Maxer.mp4).

A productionized research-and-analytics web app for **portfolio managers**. Ask the web a
question and get an answer where **every claim traces to a real, retrieved source**, or
upload a portfolio and get **code-computed** risk/return analytics with interactive charts.
In finance, precise and verifiable data is critical — so nothing here is improvised by the
model: citations are validated against what Tavily actually returned, and every metric is
computed in code, never estimated by the LLM.

This started as the take-home's [`legacy/starter_agent.py`](legacy/starter_agent.py) — a
bare agent that searched the web but couldn't prove its answers (it emitted citation-like
markers that pointed at nothing). It grew into a deployed, validated, observable product:
grounded citations, a portfolio-analytics tool, a single-page web UI, optional voice input,
and a Vercel deployment. For the approach and the value it creates, see the
[**Technical Statement**](Technical%20Statement.md); the full build arc is in
[`docs/improvements.md`](docs/improvements.md) and
[`docs/developmental_stages.md`](docs/developmental_stages.md).

## What this project built

The assignment ([`docs/2606_Tavily_FDE_TakeHomeAssignment.md`](docs/2606_Tavily_FDE_TakeHomeAssignment.md))
lists example directions for improving the starter agent. This project deliberately spans
**all of them**, each grounded in working code rather than prose:

| Direction | What this project built |
|---|---|
| **Adapt to a specific customer workflow** | Reframed from a generic search CLI into *Intelligence for Portfolio Managers*: a **Portfolio mode** ingests real holdings (CSV / Excel / `ticker,weight` or dollar-amount columns → weights from market value) and returns code-computed analytics — annualized return, volatility, Sharpe, Sortino, max drawdown, VaR/CVaR — with interactive charts. [`lib/portfolio_tool.py`](lib/portfolio_tool.py) · [`lib/quant.py`](lib/quant.py) · [design](docs/portfolio_analysis_design.md) |
| **Add a useful integration** | **yfinance** (live prices), **Plotly** (interactive charts), **ElevenLabs** speech-to-text for voice input (server-side; key never reaches the browser), and a **Vercel** deployment serving the whole app from one WSGI entrypoint — no extra web framework. |
| **Improve retrieval quality** | The Tavily tool is wrapped to expose `search_depth`, `time_range`, `include_domains`, and a finance-aware `topic` (`general` / `news` / `finance`), and to **dedupe results by normalized URL** — fixing the duplicate-result bug the starter exhibited. [`tavily_maxer.py`](tavily_maxer.py) |
| **Improve source handling & citations** | A `SourceRegistry` assigns **code-owned** source IDs (never the model's); `response_format=ResearchAnswer` forces structured `cited_source_ids`; `validate_citations` checks every `[n]` resolves to a real retrieved source; `【n】` markers are normalized to `[n]` in code; a bounded **repair loop** feeds errors back. Extended to analytics: `referenced_metric_ids` / `referenced_chart_ids` are validated, so the model **cannot fabricate a number or a chart**. |
| **Add an evaluation loop** | A golden question set spanning `simple_factual`, `recency_sensitive`, `comparison_synthesis`, `adversarial_no_source`, and `niche_factual`, run by [`evals/run_samples.py`](evals/run_samples.py) into a committed [`evals/dataset.md`](evals/dataset.md) recording the deterministic citation-validity pass rate and latency per run — the same validation used in production. |
| **Context-engineering improvement** | Sources are handed to the model **pre-labeled** with their code-owned IDs (`[n] Title — url — snippet`) so it cites concrete retrieved items instead of inventing markers; the structured schema + repair turn keep generation grounded; the registry pattern is generalized to analytics so the model references computed artifacts by `[metric:k]` / `[chart:j]` ID rather than restating values. |
| **Improve observability / debuggability** | Every run appends a structured line to [`logs/runs.jsonl`](logs/runs.jsonl) (question, tool calls, latency, validation outcome) — inspectable with no external service. When `LANGSMITH_TRACING=true`, runs are also traced in **LangSmith** with citation-validity attached as run feedback. The web UI surfaces a ✓/✗ badge plus source count, searches, latency, and model. |

## Tech stack

- **Python** — agent, analytics, and web server (standard-library `http.server`; no web
  framework, so the dependency footprint stays small).
- **Tavily** — web search; results are deduped and assigned code-owned source IDs so the
  model can only cite things that were actually retrieved.
- **LangChain** (`create_agent`, structured output) + **Nebius** — the ReAct agent loop and
  LLM synthesis, with a structured `ResearchAnswer` schema that forces inline `[n]`
  citations.
- **NumPy / pandas / yfinance / Plotly** — the portfolio engine: live prices, in-house
  risk/return quant (annualized return, volatility, Sharpe, etc.), and interactive charts.
  See [`docs/portfolio_analysis_design.md`](docs/portfolio_analysis_design.md).
- **ElevenLabs** — optional speech-to-text for voice input (server-side, key never reaches
  the browser).
- **Vercel** — hosting; the whole app deploys from one WSGI entrypoint (`webapp:app`).
- **LangSmith** — optional tracing/observability, with a local `logs/runs.jsonl` fallback.

## How to use

Open the [live app](https://tavily-take-home.vercel.app/) (or run it locally — see
[`docs/setup.md`](docs/setup.md)).

### 1. Research mode

Ask any question and get a grounded, cited answer with a ✓/✗ validation badge and a numbered
Sources panel. Try the sample question in [`Demo/research.txt`](Demo/research.txt):

> Pick top 10 best ETFs for investing in AI in 2026.

Paste it into the box, hit **Search**, and click any `[n]` citation chip to jump to its
source.

![Research result: a grounded answer with a numbered Sources panel and an "all citations verified" validation badge](Demo/research_result.png)

### 2. Portfolio mode

Switch to **Portfolio**, upload [`Demo/AAPL_SPCX.csv`](Demo/AAPL_SPCX.csv) (a holdings file
with ticker and dollar amounts — AAPL and SPCX; weights are derived from market value), and
enter a prompt like:

> Analyse my portfolio over the last 5 years

You get code-computed headline metrics (return, volatility, Sharpe), interactive charts, and
a validation badge confirming every number was computed — not estimated by the model.

![Portfolio result: cumulative performance, drawdown, and return-distribution charts beside a metrics panel (return, volatility, Sharpe, Sortino, max drawdown, VaR/CVaR)](Demo/portfolio_result.png)

### 3. Voice input (optional)

Click the 🎤 mic to **dictate** your question instead of typing. The browser records audio,
the server transcribes it via **ElevenLabs** speech-to-text, and the transcript drops into
the box for you to review and run. **No setup needed to use it on the live app** — just
allow mic access when prompted. (The ElevenLabs key lives on the server; only someone
deploying their own copy configures it — see [`docs/setup.md`](docs/setup.md).)

## Setup & deployment

Running locally, the required API keys, tests, and deploying your own copy to Vercel are all
covered in **[`docs/setup.md`](docs/setup.md)**.

## Project layout

| Path | What |
|---|---|
| [`Technical Statement.md`](Technical%20Statement.md) | Approach, thought process, and the value created. |
| [`tavily_maxer.py`](tavily_maxer.py) | The research agent — search, structured citations, validation, tracing. |
| [`webapp.py`](webapp.py) | The web app: WSGI `app` (Vercel entrypoint) + inlined UI + local dev server. |
| [`lib/`](lib/) | Supporting modules: portfolio parsing, quant, charts, market data, artifacts. |
| [`docs/`](docs/) | [setup](docs/setup.md) · [design/rationale](docs/improvements.md) · [dev log](docs/developmental_stages.md) · [portfolio design](docs/portfolio_analysis_design.md) |
| [`Demo/`](Demo/) | Sample inputs for the two modes. |
| [`tests/`](tests/), [`evals/`](evals/) | Offline unit tests and live sample/eval runs. |
| [`legacy/`](legacy/) | The original `starter_agent.py` this project grew from. |
