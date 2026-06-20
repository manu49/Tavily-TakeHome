# Setup & Deployment

Setup, local run, tests, and Vercel deployment for **Tavily Maxer**. For the product
overview and how to use it, see the [README](../README.md).

## API keys

Two keys are required (free tiers available):

1. **Tavily** — https://app.tavily.com  (web search)
2. **Nebius** — https://tokenfactory.nebius.com  (LLM inference)

Put them in `.env.local` at the repo root (or export them into the environment):

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

Optional — for hosted tracing, set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and
`LANGSMITH_PROJECT`. Without them, runs still log locally to `logs/runs.jsonl`.

## Run the web app (local)

```bash
uv run webapp.py                 # open http://127.0.0.1:8000
uv run webapp.py --port 9000     # custom port
uv run webapp.py --host 0.0.0.0  # expose on your network
```

The page is a single self-contained file: the HTML is inlined in
[`webapp.py`](../webapp.py), served by the Python standard-library `http.server` — no web
framework. It warns at startup if any key is missing.

Endpoints: `GET /` (the page), `POST /api/ask` (`{"question": "..."}` → JSON result),
`POST /api/transcribe` (raw audio bytes → `{"text": "..."}`, voice input), `GET /healthz`.

## Run the CLI

```bash
uv run tavily_maxer.py "What changed in the AI search market this year?"
```

Streams the agent's tool calls, then prints the answer, a numbered Sources panel, and a
✓/✗ citation-validation badge — the same `run_query()` path the web app and evals use.

## Tests & evals

```bash
uv run pytest                    # offline unit tests (no API calls)
uv run evals/run_samples.py      # live sample runs against the golden question set
```

## Deploy to Vercel

This repo is already configured for Vercel. Vercel's native Python runtime serves the whole
deployment from one WSGI app (`app` in [`webapp.py`](../webapp.py)), which handles both the
page (`GET /`) and the API endpoints using only the standard library. The page HTML is
inlined, so there's no external asset to bundle.

| File | Role |
|---|---|
| [`webapp.py`](../webapp.py) | The WSGI `app` (Vercel entrypoint) + inlined page HTML + the local dev server. |
| [`pyproject.toml`](../pyproject.toml) | `[tool.vercel] entrypoint = "webapp:app"` tells Vercel which app to run. |
| [`requirements.txt`](../requirements.txt) | Dependencies Vercel installs. |

**Steps:**

1. **Push to GitHub** (Vercel deploys from a Git repo):
   ```bash
   git add . && git commit -m "Deploy web app" && git push
   ```
2. **Import the repo** at [vercel.com/new](https://vercel.com/new) → pick this repository →
   **Import**. Leave the framework preset as **Other**; no build command is needed.
3. **Add environment variables** (Project → Settings → Environment Variables), for all
   environments:
   - `TAVILY_API_KEY` — your `tvly-...` key
   - `NEBIUS_API_KEY` — your Nebius key
   - *(optional)* `ELEVENLABS_API_KEY` — enables the 🎤 voice-input button
   - *(optional)* `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
4. **Deploy.** Vercel builds and gives you a `https://<project>.vercel.app` URL. If you add
   env vars after the first deploy, trigger a redeploy so they take effect.

Or deploy from the CLI:

```bash
npm i -g vercel
vercel            # first run links/creates the project (preview deploy)
vercel env add TAVILY_API_KEY      # repeat for NEBIUS_API_KEY, ELEVENLABS_API_KEY
vercel --prod     # production deploy
```

**Serverless notes:** the function runs with `log=False` (Vercel's filesystem is read-only
outside `/tmp`, so `logs/runs.jsonl` is skipped — LangSmith tracing still works via env
vars) and `repair_attempts=1` to keep latency bounded. Cold starts take a few seconds
because the LangChain stack is heavy; subsequent requests are warm. If a cold-start query
ever hits the function time limit, raise it under Project → Settings → Functions →
**Function Max Duration**.

### Why not GitHub Pages?

GitHub Pages serves **static files only**. This app needs a running Python backend: it calls
Tavily and Nebius with **secret keys** (which must never ship to the browser), and the agent
logic, portfolio analytics, and citation validation all run server-side. Use a host that
runs Python and holds secret env vars — Vercel (configured here), Render, Railway, or Fly.io.
