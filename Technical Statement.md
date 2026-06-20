# Technical Statement — Tavily Maxer

## Business context

Portfolio managers live on two jobs: **researching new investment opportunities** and
**analyzing the portfolios they already hold**. Both are revenue-sensitive and
time-critical — an allocation decision moves real money, and acting on a stale or fabricated
fact is expensive. So the bar isn't "a helpful answer"; it's **precise, verifiable sourcing
from the live web**, plus analytics they can trust. Tavily Maxer targets exactly that
workflow: sourced research *and* code-computed portfolio analytics, where every claim and
every number is auditable rather than improvised.

## The problem

The starter agent could search the web but couldn't *prove* anything it said. In testing it
emitted citation-like markers (`【0†L8-L15】`) that pointed at nothing and returned duplicate
results, with no way to inspect what was actually retrieved. For any serious user — and
especially in finance, where a wrong number is worse than no number — "it searches the web"
is not a credible pitch. The credible pitch is: *every claim traces to a real, retrieved
source, and you can verify it.* That conviction drove every decision.

## Approach

I took the "improve an existing application" path and aimed it at a concrete customer:
**portfolio managers**, who need fast, sourced market research *and* hard analytics on their
holdings. The result, **Tavily Maxer**, is a deployed web app with two modes — grounded web
research and code-computed portfolio analytics — built so that nothing user-facing is ever
improvised by the model.

The core architectural idea is **code-owned grounding**. When Tavily returns results, a
`SourceRegistry` dedupes them by normalized URL (fixing the starter's duplicate bug) and
assigns each a stable integer ID *in code, not by the model*. The model receives sources
pre-labeled (`[n] Title — url — snippet`) and answers through a structured
`response_format=ResearchAnswer` schema that forces an explicit `cited_source_ids` list.
After generation, a deterministic `validate_citations` pass checks that every `[n]` marker
resolves to a real retrieved source; a bounded repair loop feeds any violation back to the
model. Because IDs are code-owned and validated, a passing answer contains **zero
hallucinated citations by construction** — not "looks plausible," but provably grounded.

I then generalized that same pattern to analytics. The portfolio tool computes risk/return
metrics (annualized return, volatility, Sharpe, Sortino, max drawdown, VaR/CVaR) and charts
in NumPy/pandas, registers each with a code-owned ID, and the model references them by
`[metric:k]` / `[chart:j]` — validated identically. So the model **cannot fabricate a number
or embed a chart that was never built**. Every figure a portfolio manager sees was computed,
never estimated by an LLM.

## Thought process and trade-offs

A few deliberate choices:

- **Determinism over trust.** Anywhere correctness matters (citations, metrics) I used a
  code check, not a prompt. LLMs are the synthesis layer; the guarantees live in Python.
- **Less is more.** The web app runs on the Python standard-library `http.server` with the
  page HTML inlined — no web framework — so the dependency footprint stays small and the
  whole thing deploys to Vercel from one WSGI entrypoint. I replaced SciPy with an in-house
  normal distribution to keep the serverless bundle under Vercel's size limit.
- **Retrieval tuned for the domain.** The Tavily tool exposes `search_depth`, `time_range`,
  `include_domains`, and a finance-aware `topic` (`general` / `news` / `finance`), so the
  agent can pull recent, on-domain sources rather than generic blue links.
- **Pragmatism in structure.** When I reorganized the repo into `lib/`, I kept bare-name
  imports working via a one-line `sys.path` shim rather than a disruptive package rewrite —
  small change, no churn to the deploy entrypoint or documented commands.

## Spanning the brief

The work deliberately covers all of the assignment's example directions, each grounded in
working code: a **customer workflow** (portfolio analytics), **integrations** (yfinance,
Plotly, ElevenLabs speech-to-text for voice input, Vercel), **retrieval quality** (tuned
Tavily params + URL dedupe), **source handling and citations** (the registry/validation/
repair stack), an **evaluation loop**, a **context-engineering** improvement (pre-labeled,
ID-addressable sources and artifacts), and **observability**.

The **eval loop** runs a golden question set spanning categories — simple factual,
recency-sensitive, comparison/synthesis, an adversarial no-good-source case, and niche
factual — and records the deterministic citation-validity pass rate and latency per run into
a committed dataset, using the *same* validation that runs in production. For
**observability**, every run appends a structured line to `logs/runs.jsonl` (question, tool
calls, latency, validation outcome), and with one env var it also traces to **LangSmith**
with the citation-validity result attached as run feedback — so a failed run is filterable,
not just a console print.

## Value created

**Business.** It turns "AI that searches" into "AI you can audit." A portfolio manager gets
sourced research with one-click-verifiable citations and code-computed analytics on their own
holdings — in a browser, with optional voice input — without trusting a model's arithmetic.
In a regulated, precision-critical field, that verifiability is the difference between a demo
and a tool someone would actually use. It's deployed and live, not a notebook.

**Technical.** The grounding architecture is reusable: a code-owned registry + structured
output + deterministic validation + bounded repair is a general recipe for any
retrieval-augmented system that needs provable, not plausible, grounding — and it extends
cleanly from web citations to computed artifacts. Paired with an eval loop and
industry-standard tracing, the system is inspectable end to end.

The guiding principle throughout: **let the model synthesize, but never let it be the source
of truth.**
