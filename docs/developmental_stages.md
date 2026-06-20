# Developmental Stages — Issues & Fixes

A short log of problems hit while building the web demo + portfolio analysis, and the idea
behind each fix. Newest at the bottom.

## Deployment (Vercel)

| # | Issue | Fix |
|---|-------|-----|
| 1 | Build failed: *"No python entrypoint found"* — Vercel's native Python detection found the two `typer` `app` objects and couldn't choose. | Serve the whole app from one stdlib **WSGI `app`** in `webapp.py`; point Vercel at it with `[tool.vercel] entrypoint = "webapp:app"`. |
| 2 | Build kept using an **old commit** (`c6d286d`). | "Redeploy" reuses the original commit — push a new commit (or deploy latest) instead of redeploying the failed one. |
| 3 | Build failed: `functions` pattern `webapp.py` "doesn't match any Serverless Functions inside the api directory". | Drop the `vercel.json` `functions` block (api-only); **inline the page HTML** into `webapp.py` so there's no external asset to bundle. |
| 4 | Runtime: `ModuleNotFoundError: No module named 'pandas'`. | Vercel's native runtime installs only `[project.dependencies]` (not `requirements.txt` or extras) — move the portfolio stack into the **main dependencies**. |
| 5 | Risk of exceeding Vercel's **250 MB** serverless bundle. | Drop **scipy** (~100 MB); replace its normal inverse-CDF/PDF with an in-house **Acklam** approximation in `quant.py`. Keep Plotly (interactive charts ride inline in the JSON response — no storage). |

## Agent robustness

| # | Issue | Fix |
|---|-------|-----|
| 6 | *"Agent did not produce a structured ResearchAnswer"* when gpt-oss answered in prose (e.g. SpaceX). | `coerce_research_answer()` — salvage the final prose into a `ResearchAnswer` (lifting `[n]` markers) instead of crashing; still validated. |
| 7 | Validation failed on full-width **`【1】`** citation markers (gpt-oss reverting to its pretrained style); the repair turn often didn't fix it. | `normalize_citation_markers()` — deterministically rewrite `【n】`/`【1, 3】` → `[n]`/`[1][3]` before validation. Structured markers like `【1†L8-L15】` stay flagged. |

## Portfolio / quant correctness

| # | Issue | Fix |
|---|-------|-----|
| 8 | Upload with a **"Dollar Amount"** column came out 50/50 equal-weight (column ignored). | Recognize market-value/dollar columns and derive weights **proportional to value** (→ 98.68% / 1.32%); add a `value` field; auto-detect CSV delimiter (tab/comma). |
| 9 | Model hallucinated security identities (called SPCX "an S&P 500 ETF") and "equal-weight". | Prompt grounding: report the **actual** weights, and don't state a ticker's name/type unless confirmed via `tavily_search`. |

## Web UI

| # | Issue | Fix |
|---|-------|-----|
| 10 | Portfolio sections stacked vertically; overview too large. | Dashboard layout: compact, scrollable **overview on top**, then **charts + metrics side by side**. |
| 11 | Plotly charts **overflowed** into the metrics column. | Charts were first drawn while the panel was `display:none` (wrong width) — re-fit with `Plotly.Plots.resize()` once visible + on resize; cells get `min-width:0` / `overflow:hidden`. |
| 12 | Markdown **tables dumped as raw `\| ... \|` text** in the answer. | Extend the page's mini-markdown renderer to parse tables (+ horizontal rules). |
| 13 | Headline figures duplicated / drifting in the overview prose. | Render a small **Metric \| Value** table at the top of the overview built directly from the code-computed metrics (no LLM). |

## Throughline

Every fix favors **deterministic, code-owned, testable** behavior over asking the LLM to
self-correct: weights, metrics, charts, and citation/marker handling are computed or
normalized in code and validated, so the model can reference artifacts but never fabricate
them.
