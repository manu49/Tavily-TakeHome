# Implementation Plan — Grounded Citations + Observability

## Motivation

The starter agent answers research questions but cannot prove its answers. In testing, it
emitted citation-like markers (`【0†L8-L15】`) that pointed at nothing, and returned duplicate
search results from a single query with no indication of what was actually retrieved or why
the model said what it said. For a customer evaluating Tavily in a research/support-agent
context, the credible pitch isn't "it searches the web" — it's "every claim traces to a real,
retrieved source, and you can inspect exactly how." This plan makes citations structurally
enforced (not improvised by the model) and makes every run inspectable via tracing, with a
small eval suite to catch regressions in both.

## Architecture

```
question
   │
   ▼
create_agent (ReAct loop)
   │   tool calls ──▶ SourceRegistry-wrapped TavilySearch
   │                      │  dedupes by normalized URL
   │                      │  assigns stable numeric IDs (code-owned, not model-owned)
   │                      ▼
   │                  registry: {id → {title, url}}
   │
   ▼ (response_format=ResearchAnswer)
structured final answer: { answer_markdown, cited_source_ids[] }
   │
   ▼
validate_citations(answer, registry)
   │  every cited id exists in registry → pass
   │  every registry url was actually returned by Tavily → pass (guaranteed by construction)
   ▼
render (CLI: answer + numbered Sources panel + validation badge)
   │
   ▼
trace (LangSmith run + local JSONL fallback; validation result attached as run feedback)
```

## Components

### 1. `SourceRegistry` + wrapped Tavily tool (`sources.py`)
A thin wrapper around `TavilySearch` that intercepts results before they reach the model:
- Normalizes and dedupes by URL (fixes the duplicate-result issue observed in testing).
- Assigns each unique result a stable integer ID, owned by code, not the LLM.
- Returns results to the model pre-labeled (`[3] Title — url — snippet`) so the model has
  something concrete to cite instead of inventing markers.
- Registry persists for the duration of one run and is the single source of truth for
  validation.

### 2. Structured response schema (`schema.py`)
`langchain.agents.create_agent` natively supports `response_format` (verified against the
installed `langchain` 1.x — it adds a structured-output step via tool-calling after the ReAct
loop finishes, no custom graph wiring needed):

```python
class Source(BaseModel):
    id: int
    title: str
    url: str

class ResearchAnswer(BaseModel):
    answer: str               # markdown, inline [n] citation markers
    cited_source_ids: list[int]
```

`create_agent(..., response_format=ResearchAnswer)` forces the final turn through this schema
instead of free-text, eliminating the class of bug where citation markers are stylistic
guesses.

### 3. Validation layer (`validation.py`)
Deterministic, cheap, runs after every response:
- Every id in `cited_source_ids` (and every `[n]` marker found in `answer`) must exist in the
  `SourceRegistry`.
- Every source the registry holds was, by construction, an actual Tavily result — so a passing
  check means zero hallucinated URLs, not just "looks plausible."
- On failure: surface a visible warning badge in the CLI and tag the trace — don't silently
  accept it (matches existing error-handling gaps in the starter agent, where failures were
  invisible or fatal).

### 4. Observability (`tracing.py`)
- LangSmith via env vars only (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) —
  near-zero code, auto-instruments the agent graph, tool calls, and token usage.
- Attach the validation result as run feedback (`client.create_feedback(run_id,
  key="citation_validity", score=0|1)`) so a failed trace is filterable in the LangSmith UI,
  not just a console print.
- Local fallback: append a structured JSONL line per run (`logs/runs.jsonl`: timestamp,
  question, tool calls issued, latency, token usage, validation pass/fail) so observability
  isn't gated behind having a LangSmith account — important since this is a take-home a
  reviewer will run with their own keys.

### 5. CLI rendering (`cli.py`)
- Keep the existing streaming UX for the ReAct/tool-call phase.
- Replace the current free-text "Assistant" panel with: rendered `answer` markdown + a
  numbered "Sources" panel (id, title, clickable URL) + a small ✓/✗ validation badge.

## Evaluation loop (`evals/`)

- `evals/questions.yaml` — ~12 golden questions spanning categories: simple factual lookup,
  recency-sensitive ("this year" style — the exact case tested manually), comparison/
  multi-source synthesis, and an adversarial case with no good source (to check the model
  doesn't force a citation).
- `evals/run_evals.py` — runs the agent against each question, captures the structured
  `ResearchAnswer` + registry, and scores:
  - **Deterministic (primary):** citation validity (same check as production), non-empty
    sources, no orphaned high-relevance results ignored.
  - **LLM-as-judge (secondary, optional):** 1–5 relevance/completeness rubric via a cheap
    model call, kept separate from the deterministic gate so eval pass/fail never depends on
    another LLM's mood.
- Output: a local pass-rate report (`evals/report.json`) and, if `LANGSMITH_API_KEY` is set,
  pushed as a LangSmith dataset + evaluator run for trend tracking across changes.

## File structure

```
agent/
  config.py        # env loading, defaults
  models.py         # FixedChatNebius (Nebius streaming tool-call-index fix, already verified)
  sources.py         # SourceRegistry + wrapped TavilySearch
  schema.py            # ResearchAnswer, Source
  validation.py          # citation validator
  tracing.py               # LangSmith setup + JSONL fallback
  agent.py                   # build_agent() factory
  cli.py                       # typer CLI + rich rendering
evals/
  questions.yaml
  run_evals.py
  report.json (generated)
README.md
TECHNICAL_STATEMENT.md
```

## Build phases (fits the 4–6 hr budget)

| Phase | Time | Output |
|---|---|---|
| 1. Scaffold | 45m | Working agent loop, `FixedChatNebius`, `SourceRegistry`-wrapped tool |
| 2. Structured citations | 60m | `response_format=ResearchAnswer`, Sources panel in CLI |
| 3. Validation | 45m | Post-hoc citation check, visible pass/fail badge |
| 4. Tracing | 45m | LangSmith wiring + JSONL fallback + feedback scoring |
| 5. Evals | 60m | Golden question set, deterministic checks, pass-rate report |
| 6. Polish | 30m | README, technical statement, cleanup |

## Deliverable checklist (per assignment)

- [ ] Repo excludes `starter_agent.py`
- [ ] `TECHNICAL_STATEMENT.md` — approach, why citations+observability, business value
- [ ] Build record (this session's chat history / equivalent)
- [ ] Bonus: LangSmith tracing satisfies "industry-standard observability" callout
