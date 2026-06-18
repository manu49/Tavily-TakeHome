# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langchain>=1.0.0",
#   "langchain-nebius>=0.1.0",
#   "langchain-tavily>=0.2.0",
#   "langsmith>=0.1.0",
#   "python-dotenv>=1.0.0",
#   "rich>=13.0.0",
#   "typer>=0.12.0",
# ]
# ///
"""
webapp.py — a small landing/demo page for the tavily_maxer agent (local dev server).

Serves a single-page UI where you type a question and get back a citation-grounded
answer from the Tavily-backed research agent, plus the numbered sources it used and a
validation badge proving every citation resolves to a real retrieved source.

It reuses the exact same `run_query()` path as the CLI and the eval suite, so the demo
exercises the real agent (Tavily web search + Nebius model + citation validation), not a
mock. No web framework is added: this runs on the Python standard library `http.server`
so the project's dependency footprint is unchanged.

The page (the `PAGE` HTML constant below) and the WSGI `app` are both defined here, so the
whole demo is one self-contained file with no external asset to bundle — the local dev
server and the Vercel deployment serve the exact same front end and POST to `/api/ask`.

Run:
    uv run webapp.py            # then open http://127.0.0.1:8000
    uv run webapp.py --port 9000
    uv run webapp.py --host 0.0.0.0   # expose on your network

Requires the same two keys as the CLI (read from the environment or a .env / .env.local):
    TAVILY_API_KEY="tvly-..."
    NEBIUS_API_KEY="..."
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv

import tavily_maxer as tm

load_dotenv()
load_dotenv(".env.local", override=False)

# --------------------------------------------------------------------------------------
# Agent bridge: turn a question into a JSON-serializable result the page can render.
# --------------------------------------------------------------------------------------

def answer_question(question: str, model: str = tm.DEFAULT_MODEL) -> dict:
    """Run one question through the real agent and shape the result for the browser."""
    result = tm.run_query(question, model=model)
    return shape_result(result)


def shape_result(result: "tm.RunResult") -> dict:
    """Shared result shaping used by both this dev server and the Vercel function."""
    return {
        "question": result.question,
        "model": result.model,
        "answer": result.answer.answer,
        "cited_source_ids": sorted(set(result.answer.cited_source_ids)),
        "sources": [
            {"id": r.id, "title": r.title, "url": r.url}
            for r in result.registry.all_sources()
        ],
        "validation": {
            "valid": result.validation.valid,
            "errors": result.validation.errors,
        },
        "tool_calls": result.tool_calls,
        "latency_seconds": round(result.latency_seconds, 2),
    }


def load_page() -> bytes:
    return PAGE.encode("utf-8")


# --------------------------------------------------------------------------------------
# WSGI app — the entrypoint used on Vercel (see [tool.vercel] in pyproject.toml).
#
# Vercel's native Python runtime serves the whole deployment from a single WSGI/ASGI
# `app`, so this one callable handles both the static page (GET /) and the agent
# endpoint (POST /api/ask) — the same routes the local dev server below exposes. It is
# pure standard library: no framework, no extra dependency. Serverless specifics match
# api-style constraints: log=False (read-only filesystem outside /tmp; LangSmith tracing
# still works via env vars) and repair_attempts=1 (stay inside the function maxDuration).
# --------------------------------------------------------------------------------------

_STATUS_LINES = {
    200: "200 OK", 400: "400 Bad Request", 404: "404 Not Found",
    405: "405 Method Not Allowed", 500: "500 Internal Server Error",
}


def _wsgi_bytes(start_response, status: int, body: bytes, content_type: str):
    start_response(
        _STATUS_LINES[status],
        [("Content-Type", content_type), ("Content-Length", str(len(body)))],
    )
    return [body]


def _wsgi_json(start_response, status: int, payload: dict):
    return _wsgi_bytes(start_response, status, json.dumps(payload).encode("utf-8"), "application/json")


def app(environ, start_response):
    """WSGI entrypoint for Vercel (and any WSGI host)."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/") or "/"

    if method == "GET" and path in ("/", "/index.html"):
        return _wsgi_bytes(start_response, 200, load_page(), "text/html; charset=utf-8")
    if method == "GET" and path == "/healthz":
        return _wsgi_json(start_response, 200, {"ok": True})
    if path != "/api/ask":
        return _wsgi_json(start_response, 404, {"error": "not found"})
    if method != "POST":
        return _wsgi_json(start_response, 405, {"error": "POST a JSON body {\"question\": \"...\"}."})

    try:
        size = int(environ.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        size = 0
    raw = environ["wsgi.input"].read(size) if size > 0 else b""
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return _wsgi_json(start_response, 400, {"error": "invalid JSON body"})

    question = (payload.get("question") or "").strip()
    if not question:
        return _wsgi_json(start_response, 400, {"error": "Please enter a question."})

    try:
        result = tm.run_query(question, model=tm.DEFAULT_MODEL, log=False, repair_attempts=1)
        return _wsgi_json(start_response, 200, shape_result(result))
    except Exception as exc:
        traceback.print_exc()
        return _wsgi_json(start_response, 500, {"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------------------
# HTTP layer (local dev server)
# --------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    # Quieter logging: one line per request is enough for a demo.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[webapp] {self.address_string()} - {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, load_page(), "text/html; charset=utf-8")
        elif self.path == "/healthz":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        question = (payload.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "Please enter a question."})
            return

        try:
            self._send_json(200, answer_question(question))
        except Exception as exc:  # surface the failure to the page instead of a blank 500
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def check_keys() -> list[str]:
    return [name for name in ("TAVILY_API_KEY", "NEBIUS_API_KEY") if not os.getenv(name)]


def main() -> None:
    parser = argparse.ArgumentParser(description="tavily_maxer demo web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    missing = check_keys()
    if missing:
        print(f"[webapp] WARNING: missing {', '.join(missing)} — searches will fail until set.")
        print("[webapp]   export TAVILY_API_KEY='tvly-...'  (https://app.tavily.com)")
        print("[webapp]   export NEBIUS_API_KEY='...'        (https://tokenfactory.nebius.com)")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[webapp] tavily_maxer demo running at {url}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webapp] shutting down.")
        server.shutdown()


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>tavily_maxer — grounded web research</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #121a30;
    --panel-2: #0e1528;
    --border: #233055;
    --text: #e8ecf6;
    --muted: #93a0bd;
    --accent: #4f7cff;
    --accent-2: #36d6c3;
    --good: #2bd576;
    --bad: #ff6b6b;
    --chip: #1a2444;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: radial-gradient(1200px 700px at 15% -10%, #16224a, transparent),
                radial-gradient(1000px 600px at 100% 0%, #122044, transparent),
                var(--bg);
    color: var(--text);
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    min-height: 100vh;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 56px 20px 96px; }
  header { text-align: center; margin-bottom: 36px; }
  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 6px 14px; margin-bottom: 20px; background: var(--panel-2);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-2); box-shadow: 0 0 10px var(--accent-2); }
  h1 { font-size: 40px; line-height: 1.1; margin: 0 0 12px; letter-spacing: -0.02em; }
  h1 .grad { background: linear-gradient(90deg, var(--accent), var(--accent-2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .sub { color: var(--muted); font-size: 17px; max-width: 620px; margin: 0 auto; }
  form {
    margin-top: 28px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; padding: 14px; display: flex; gap: 10px;
    box-shadow: 0 20px 50px -20px rgba(0,0,0,.6);
  }
  textarea {
    flex: 1; resize: none; background: transparent; color: var(--text);
    border: none; outline: none; font: inherit; padding: 8px 6px; min-height: 26px; max-height: 180px;
  }
  textarea::placeholder { color: #5e6b8a; }
  button.ask {
    align-self: flex-end; background: linear-gradient(90deg, var(--accent), #6a5cff);
    color: white; border: none; border-radius: 10px; padding: 11px 20px;
    font-weight: 600; font-size: 15px; cursor: pointer; transition: transform .05s ease, opacity .2s;
    white-space: nowrap;
  }
  button.ask:hover { transform: translateY(-1px); }
  button.ask:disabled { opacity: .5; cursor: default; transform: none; }
  .examples { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .chip {
    font-size: 13px; color: var(--muted); background: var(--chip);
    border: 1px solid var(--border); border-radius: 999px; padding: 6px 12px; cursor: pointer;
    transition: color .15s, border-color .15s;
  }
  .chip:hover { color: var(--text); border-color: var(--accent); }

  .status { margin-top: 28px; text-align: center; color: var(--muted); display: none; }
  .status.show { display: block; }
  .spinner {
    width: 18px; height: 18px; border-radius: 50%;
    border: 2px solid var(--border); border-top-color: var(--accent);
    display: inline-block; vertical-align: -3px; margin-right: 8px;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .results { margin-top: 32px; display: none; }
  .results.show { display: block; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px 24px; margin-bottom: 18px;
  }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 14px; }
  .answer { font-size: 16.5px; line-height: 1.7; }
  .answer h1, .answer h2, .answer h3 { font-size: 19px; margin: 18px 0 8px; }
  .answer p { margin: 0 0 12px; }
  .answer ul, .answer ol { margin: 0 0 12px; padding-left: 22px; }
  .answer code { background: #0c1430; padding: 1px 6px; border-radius: 5px; font-size: 14px; }
  .cite {
    display: inline-block; font-size: 12px; font-weight: 700; line-height: 1;
    color: var(--accent-2); background: rgba(54,214,195,.12);
    border: 1px solid rgba(54,214,195,.3); border-radius: 6px;
    padding: 2px 6px; margin: 0 2px; cursor: pointer; text-decoration: none; vertical-align: 1px;
  }
  .cite:hover { background: rgba(54,214,195,.22); }

  .src { display: flex; gap: 12px; padding: 12px 0; border-top: 1px solid var(--border); }
  .src:first-of-type { border-top: none; }
  .src .num {
    flex: none; width: 26px; height: 26px; border-radius: 7px; font-size: 13px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    color: var(--accent-2); background: rgba(54,214,195,.12); border: 1px solid rgba(54,214,195,.3);
  }
  .src .meta { min-width: 0; }
  .src .title { font-weight: 600; }
  .src a { color: var(--accent); text-decoration: none; font-size: 13px; word-break: break-all; }
  .src a:hover { text-decoration: underline; }
  .src.target { animation: flash 1.4s ease; }
  @keyframes flash { 0% { background: rgba(79,124,255,.18); } 100% { background: transparent; } }

  .badge-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .vbadge { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px; padding: 8px 14px; border-radius: 10px; }
  .vbadge.ok { color: var(--good); background: rgba(43,213,118,.1); border: 1px solid rgba(43,213,118,.3); }
  .vbadge.fail { color: var(--bad); background: rgba(255,107,107,.1); border: 1px solid rgba(255,107,107,.3); }
  .stat { font-size: 13px; color: var(--muted); }
  .stat b { color: var(--text); font-weight: 600; }

  .error { color: var(--bad); background: rgba(255,107,107,.08); border: 1px solid rgba(255,107,107,.3); border-radius: 12px; padding: 16px 18px; }
  footer { text-align: center; color: #4f5d7e; font-size: 13px; margin-top: 40px; }
  footer a { color: var(--muted); }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="badge"><span class="dot"></span> Powered by Tavily web search · Nebius · LangChain</div>
      <h1>tavily<span class="grad">_maxer</span></h1>
      <p class="sub">Ask anything. Get a research answer where <b>every claim traces to a real
      retrieved source</b> — citations are validated, not improvised.</p>
    </header>

    <form id="form">
      <textarea id="q" placeholder="What changed in the AI search market this year?" rows="1"></textarea>
      <button class="ask" id="ask" type="submit">Search</button>
    </form>

    <div class="examples" id="examples">
      <span class="chip">What changed in the AI search market this year?</span>
      <span class="chip">Who won the most recent Nobel Prize in Physics?</span>
      <span class="chip">Compare the latest flagship phones from Apple and Google.</span>
      <span class="chip">What are the newest features in Python 3.13?</span>
    </div>

    <div class="status" id="status"><span class="spinner"></span><span id="statusText">Searching the web and grounding the answer…</span></div>

    <div class="results" id="results">
      <div class="card">
        <h2>Answer</h2>
        <div class="answer" id="answer"></div>
      </div>
      <div class="card" id="sourcesCard">
        <h2>Sources</h2>
        <div id="sources"></div>
      </div>
      <div class="card">
        <h2>Verification</h2>
        <div class="badge-row">
          <span id="vbadge"></span>
          <span class="stat" id="stats"></span>
        </div>
      </div>
    </div>

    <div class="results" id="errorBox">
      <div class="error" id="errorText"></div>
    </div>

    <footer>Reuses the same validated agent path as the CLI · sources verified against Tavily results</footer>
  </div>

<script>
const form = document.getElementById('form');
const q = document.getElementById('q');
const askBtn = document.getElementById('ask');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('statusText');
const results = document.getElementById('results');
const errorBox = document.getElementById('errorBox');
const errorText = document.getElementById('errorText');
const answerEl = document.getElementById('answer');
const sourcesEl = document.getElementById('sources');
const sourcesCard = document.getElementById('sourcesCard');
const vbadge = document.getElementById('vbadge');
const stats = document.getElementById('stats');

// auto-grow textarea
q.addEventListener('input', () => { q.style.height = 'auto'; q.style.height = q.scrollHeight + 'px'; });

document.getElementById('examples').addEventListener('click', (e) => {
  if (e.target.classList.contains('chip')) { q.value = e.target.textContent; q.focus(); q.dispatchEvent(new Event('input')); }
});

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Minimal, safe markdown → HTML: headings, bold/italic, inline code, lists, paragraphs.
// Citation markers [n] are turned into clickable chips that jump to the source.
function renderMarkdown(md) {
  const lines = escapeHtml(md).split('\n');
  let html = '', inList = null;
  const inline = (t) => t
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(\d+)\]/g, '<a class="cite" href="#src-$1" data-id="$1">$1</a>');
  const closeList = () => { if (inList) { html += `</${inList}>`; inList = null; } };
  for (let raw of lines) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }
    let m;
    if ((m = line.match(/^(#{1,3})\s+(.*)$/))) { closeList(); const n = m[1].length; html += `<h${n}>${inline(m[2])}</h${n}>`; }
    else if ((m = line.match(/^[-*]\s+(.*)$/))) { if (inList !== 'ul') { closeList(); html += '<ul>'; inList = 'ul'; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\d+\.\s+(.*)$/))) { if (inList !== 'ol') { closeList(); html += '<ol>'; inList = 'ol'; } html += `<li>${inline(m[1])}</li>`; }
    else { closeList(); html += `<p>${inline(line)}</p>`; }
  }
  closeList();
  return html;
}

function renderSources(sources, cited) {
  if (!sources.length) { sourcesCard.style.display = 'none'; return; }
  sourcesCard.style.display = '';
  sourcesEl.innerHTML = sources.map(s => `
    <div class="src" id="src-${s.id}">
      <div class="num">${s.id}</div>
      <div class="meta">
        <div class="title">${escapeHtml(s.title)}</div>
        <a href="${encodeURI(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.url)}</a>
      </div>
    </div>`).join('');
}

// Clicking a citation chip scrolls to and flashes its source row.
answerEl.addEventListener('click', (e) => {
  const a = e.target.closest('.cite');
  if (!a) return;
  e.preventDefault();
  const row = document.getElementById('src-' + a.dataset.id);
  if (row) { row.scrollIntoView({ behavior: 'smooth', block: 'center' }); row.classList.remove('target'); void row.offsetWidth; row.classList.add('target'); }
});

async function ask(question) {
  errorBox.classList.remove('show');
  results.classList.remove('show');
  statusEl.classList.add('show');
  askBtn.disabled = true;
  statusText.textContent = 'Searching the web and grounding the answer…';
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));

    answerEl.innerHTML = renderMarkdown(data.answer);
    renderSources(data.sources, data.cited_source_ids);

    if (data.validation.valid) {
      vbadge.className = 'vbadge ok';
      vbadge.innerHTML = '&#10003; All citations verified against retrieved sources';
    } else {
      vbadge.className = 'vbadge fail';
      vbadge.innerHTML = '&#10007; ' + escapeHtml((data.validation.errors || []).join('; ') || 'Citation validation failed');
    }
    const searches = (data.tool_calls || []).filter(c => (c.name || '').includes('search') || (c.name || '').includes('tavily')).length;
    stats.innerHTML = `<b>${data.sources.length}</b> sources · <b>${searches}</b> web search${searches === 1 ? '' : 'es'} · <b>${data.latency_seconds}s</b> · ${escapeHtml(data.model)}`;

    statusEl.classList.remove('show');
    results.classList.add('show');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    statusEl.classList.remove('show');
    errorText.textContent = err.message;
    errorBox.classList.add('show');
  } finally {
    askBtn.disabled = false;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (question) ask(question);
});

// Cmd/Ctrl+Enter submits from the textarea.
q.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); form.requestSubmit(); }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
