# tavily_maxer

A Tavily + LangChain research agent with **grounded citations** and **observability**.
Ask a question, get a research answer where every claim traces to a real, retrieved web
source — citations are validated against what Tavily actually returned, not improvised by
the model.

The agent itself lives in [`tavily_maxer.py`](tavily_maxer.py); supporting modules (quant,
charts, portfolio, market data) are under [`lib/`](lib/) and design docs under
[`docs/`](docs/). The rationale and design are in
[`docs/improvements.md`](docs/improvements.md).

## Setup

Two API keys are required (free tiers available):

1. **Tavily** — https://app.tavily.com
2. **Nebius** — https://tokenfactory.nebius.com

Put them in `.env.local` (or export them):

```bash
TAVILY_API_KEY="tvly-..."
NEBIUS_API_KEY="..."
```

Optional — for **voice input** (the 🎤 mic button in the web demo), add an
[ElevenLabs](https://elevenlabs.io) key. Without it the mic is simply disabled; search and
portfolio analysis work unchanged.

```bash
ELEVENLABS_API_KEY="..."
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
- **Voice input (optional).** Click the 🎤 mic to dictate a question: the browser records
  audio, `POST /api/transcribe` forwards it to ElevenLabs speech-to-text (the key stays
  server-side), and the transcript drops into the box for you to review and search. Enabled
  only when `ELEVENLABS_API_KEY` is set; needs a secure context (localhost or HTTPS).
- **Verification badge + stats.** Shows ✓/✗ validation, source count, number of web
  searches, latency, and the model used.
- **Zero extra dependencies.** Built on the Python standard-library `http.server` with one
  inline HTML page — the project's dependency footprint is unchanged. Reads the same
  `TAVILY_API_KEY` / `NEBIUS_API_KEY` from the environment or `.env.local` and warns at
  startup if either is missing.

Endpoints: `GET /` (the page), `POST /api/ask` (`{"question": "..."}` → JSON result),
`POST /api/transcribe` (raw audio bytes → `{"text": "..."}`, voice input), `GET /healthz`.


### Deploy to Vercel

This repo is already set up for Vercel. Vercel's native Python runtime serves the whole
deployment from one WSGI app (`app` in `webapp.py`), which handles both the page (`GET /`)
and the agent endpoint (`POST /api/ask`) using only the standard library. The whole demo is
one self-contained file — the page HTML is inlined in `webapp.py`, so there's no external
asset for Vercel to bundle.

| File | Role |
|---|---|
| [`webapp.py`](webapp.py) | The WSGI `app` (Vercel entrypoint) + inlined page HTML + the local dev server. |
| [`pyproject.toml`](pyproject.toml) | `[tool.vercel] entrypoint = "webapp:app"` tells Vercel which app to run. |
| [`requirements.txt`](requirements.txt) | Dependencies Vercel installs. |

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
   - *(optional)* `ELEVENLABS_API_KEY` — enables the 🎤 voice-input button
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
vars) and `repair_attempts=1` to keep latency bounded. Cold starts take a few seconds
because the LangChain stack is heavy; subsequent requests are warm. If a cold-start query
ever hits the function time limit, raise it under Project → Settings → Functions →
**Function Max Duration**.

## Tests & evals

```bash
uv run pytest                    # offline unit tests (no API calls)
uv run evals/run_samples.py      # live sample runs against the golden question set
```

## Traces
https://traces.com/
