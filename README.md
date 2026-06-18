# tavily_maxer

A Tavily + LangChain research agent with **grounded citations** and **observability**.
Ask a question, get a research answer where every claim traces to a real, retrieved web
source — citations are validated against what Tavily actually returned, not improvised by
the model.

The agent itself lives in [`tavily_maxer.py`](tavily_maxer.py); the rationale and design
are in [`improvements.md`](improvements.md).

## Setup

Two API keys are required (free tiers available):

1. **Tavily** — https://app.tavily.com
2. **Nebius** — https://tokenfactory.nebius.com

Put them in `.env.local` (or export them):

```bash
TAVILY_API_KEY="tvly-..."
NEBIUS_API_KEY="..."
```

## CLI

```bash
uv run tavily_maxer.py "What changed in the AI search market this year?"
```

Streams the agent's tool calls, then prints the answer, a numbered Sources panel, and a
✓/✗ citation-validation badge.

## Web demo

[`webapp.py`](webapp.py) serves a single-page landing/demo UI: type a question, hit
**Search**, and get the grounded answer, numbered sources, and a verification badge — in
the browser.

```bash
uv run webapp.py                 # open http://127.0.0.1:8000
uv run webapp.py --port 9000     # custom port
uv run webapp.py --host 0.0.0.0  # expose on your network
```

What it does:

- **Real agent, not a mock.** `POST /api/ask` calls the same
  [`run_query()`](tavily_maxer.py) path used by the CLI and the eval suite, so every demo
  search runs live Tavily web search → Nebius synthesis → citation validation.
- **Clickable citations.** `[n]` markers in the answer become chips that scroll to and
  highlight the matching source.
- **Verification badge + stats.** Shows ✓/✗ validation, source count, number of web
  searches, latency, and the model used.
- **Zero extra dependencies.** Built on the Python standard-library `http.server` with one
  inline HTML page — the project's dependency footprint is unchanged. Reads the same
  `TAVILY_API_KEY` / `NEBIUS_API_KEY` from the environment or `.env.local` and warns at
  startup if either is missing.

Endpoints: `GET /` (the page), `POST /api/ask` (`{"question": "..."}` → JSON result),
`GET /healthz`.

### Can I host this on GitHub Pages?

**No — GitHub Pages serves static files only.** This demo needs a running Python backend
because:

- It calls the Tavily and Nebius APIs using **secret keys**, which must never be shipped to
  a static page (anyone could read them in the page source).
- The agent logic and citation validation run server-side in Python.

To deploy it for real, use a host that can run Python and hold secret environment
variables — e.g. [Vercel](https://vercel.com) (configured below),
[Render](https://render.com), [Railway](https://railway.app), or
[Fly.io](https://fly.io). GitHub Pages could host the HTML shell, but the **Search** button
would have no backend to call, so the demo wouldn't function.

### Deploy to Vercel

This repo is already set up for Vercel. The page is a static file and only the search
endpoint runs Python:

| File | Role |
|---|---|
| [`index.html`](index.html) | Static page, served at `/` (also used by the local `webapp.py`). |
| [`api/ask.py`](api/ask.py) | Python serverless function, served at `/api/ask`. Runs the real agent. |
| [`requirements.txt`](requirements.txt) | Dependencies Vercel installs for the function. |
| [`vercel.json`](vercel.json) | Sets `maxDuration` and bundles `tavily_maxer.py` with the function. |

**Steps:**

1. **Push to GitHub** (Vercel deploys from a Git repo):
   ```bash
   git add .
   git commit -m "Add Vercel-deployable web demo"
   git push
   ```
2. **Import the repo** at [vercel.com/new](https://vercel.com/new) → pick this repository →
   **Import**. Leave the framework preset as **Other**; no build command is needed.
3. **Add environment variables** (Project → Settings → Environment Variables), for all
   environments:
   - `TAVILY_API_KEY` — your `tvly-...` key
   - `NEBIUS_API_KEY` — your Nebius key
   - *(optional)* `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
4. **Deploy.** Vercel builds and gives you a `https://<project>.vercel.app` URL. If you
   added the env vars after the first deploy, trigger a redeploy so they take effect.

Or deploy from the CLI:
```bash
npm i -g vercel
vercel            # first run links/creates the project (preview deploy)
vercel env add TAVILY_API_KEY      # repeat for NEBIUS_API_KEY
vercel --prod     # production deploy
```

**Serverless notes:** the function runs with `log=False` (Vercel's filesystem is read-only
outside `/tmp`, so `logs/runs.jsonl` is skipped — LangSmith tracing still works via env
vars) and `repair_attempts=1` to stay inside the 60s `maxDuration`. Cold starts take a few
seconds because the LangChain stack is heavy; subsequent requests are warm.

## Tests & evals

```bash
uv run pytest                    # offline unit tests (no API calls)
uv run evals/run_samples.py      # live sample runs against the golden question set
```
