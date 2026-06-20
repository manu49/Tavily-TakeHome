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
import base64
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

# Supporting library modules (portfolio, quant, charts, ...) live in lib/; put it on
# sys.path so this app and tavily_maxer keep importing them by name.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import tavily_maxer as tm

load_dotenv()
load_dotenv(".env.local", override=False)

# --------------------------------------------------------------------------------------
# Agent bridge: turn a question into a JSON-serializable result the page can render.
# --------------------------------------------------------------------------------------

def shape_result(result: "tm.RunResult") -> dict:
    """Shape a RunResult into the JSON the page renders. Includes portfolio metrics and
    (interactive Plotly) chart specs when the analyze_portfolio tool ran."""
    out = {
        "question": result.question,
        "model": result.model,
        "answer": result.answer.answer,
        "cited_source_ids": sorted(set(result.answer.cited_source_ids)),
        "referenced_metric_ids": sorted(set(result.answer.referenced_metric_ids)),
        "referenced_chart_ids": sorted(set(result.answer.referenced_chart_ids)),
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
    if result.metric_registry is not None and len(result.metric_registry):
        out["metrics"] = [m.to_dict() for m in result.metric_registry.all_metrics()]
    if result.chart_registry is not None and len(result.chart_registry):
        out["charts"] = [c.to_dict() for c in result.chart_registry.all_charts()]
    return out


def process_ask(payload: dict) -> tuple[int, dict]:
    """Core request handler shared by the WSGI app and the local dev server.

    Research mode (default): a question -> grounded answer. Portfolio mode (an uploaded
    file, or portfolio_mode=true): the agent gets analyze_portfolio (+ the uploaded
    holdings) and returns metrics and interactive charts alongside the answer. Upload is
    handled in a single request (base64 in the JSON body) -- no server-side file storage,
    which is what keeps this working on Vercel's ephemeral filesystem."""
    question = (payload.get("question") or "").strip()
    file_b64 = payload.get("file_b64")
    filename = payload.get("filename") or "portfolio.csv"
    portfolio_mode = bool(payload.get("portfolio_mode") or file_b64)

    if not question and not file_b64:
        return 400, {"error": "Please enter a question (or upload a portfolio)."}

    portfolio = None
    parse_report = None
    if file_b64:
        try:
            content = base64.b64decode(file_b64)
        except Exception:
            return 400, {"error": "Could not decode the uploaded file."}
        from portfolio import parse_portfolio

        parsed = parse_portfolio(content, filename)
        parse_report = {"ok": parsed.ok, "warnings": parsed.warnings, "errors": parsed.errors}
        if not parsed.ok:
            return 400, {
                "error": "Could not read portfolio: " + "; ".join(parsed.errors),
                "parse_report": parse_report,
            }
        portfolio = parsed.portfolio
        if not question:
            question = "Analyze the risk and return of my portfolio, including the key charts."

    result = tm.run_query(
        question, model=tm.DEFAULT_MODEL, log=False, repair_attempts=1,
        enable_portfolio=portfolio_mode, portfolio=portfolio,
    )
    out = shape_result(result)
    if parse_report is not None:
        out["parse_report"] = parse_report
    return 200, out


# --------------------------------------------------------------------------------------
# Speech-to-text bridge: ElevenLabs transcription for the mic button on the page.
# --------------------------------------------------------------------------------------

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_STT_MODEL = "scribe_v1"

# Map browser MediaRecorder MIME types to a sensible upload filename extension.
_AUDIO_EXT = {
    "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4",
    "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
}


def process_transcribe(audio: bytes, content_type: str) -> tuple[int, dict]:
    """Forward recorded mic audio to ElevenLabs Speech-to-Text and return {"text": ...}.

    The ELEVENLABS_API_KEY stays server-side (same pattern as TAVILY/NEBIUS): the browser
    only ever uploads raw audio bytes, never the key. Built on stdlib urllib so the
    project's dependency footprint is unchanged."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return 500, {"error": "Voice input is unavailable: ELEVENLABS_API_KEY is not set on the server."}
    if not audio:
        return 400, {"error": "No audio received."}

    mime = (content_type or "audio/webm").split(";")[0].strip()
    ext = _AUDIO_EXT.get(mime, "webm")

    # Build a multipart/form-data body by hand (no `requests`/SDK dependency).
    boundary = uuid.uuid4().hex
    preamble = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model_id"\r\n\r\n'
        f"{ELEVENLABS_STT_MODEL}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.{ext}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    epilogue = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = preamble + audio + epilogue

    req = urllib.request.Request(
        ELEVENLABS_STT_URL,
        data=body,
        headers={
            "xi-api-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        return 502, {"error": f"ElevenLabs returned {exc.code}: {detail}"}
    except urllib.error.URLError as exc:
        return 502, {"error": f"Could not reach ElevenLabs: {exc.reason}"}

    return 200, {"text": (data.get("text") or "").strip()}


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

    if path == "/api/transcribe":
        if method != "POST":
            return _wsgi_json(start_response, 405, {"error": "POST audio bytes to /api/transcribe."})
        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError):
            size = 0
        audio = environ["wsgi.input"].read(size) if size > 0 else b""
        try:
            status, body = process_transcribe(audio, environ.get("CONTENT_TYPE", ""))
            return _wsgi_json(start_response, status, body)
        except Exception as exc:
            traceback.print_exc()
            return _wsgi_json(start_response, 500, {"error": f"{type(exc).__name__}: {exc}"})

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

    try:
        status, body = process_ask(payload)
        return _wsgi_json(start_response, status, body)
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
        if self.path == "/api/transcribe":
            length = int(self.headers.get("Content-Length", 0) or 0)
            audio = self.rfile.read(length) if length > 0 else b""
            try:
                status, body = process_transcribe(audio, self.headers.get("Content-Type", ""))
                self._send_json(status, body)
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        if self.path != "/api/ask":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        try:
            status, body = process_ask(payload)
            self._send_json(status, body)
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

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("[webapp] NOTE: ELEVENLABS_API_KEY not set — the mic / voice input will be disabled.")
        print("[webapp]   export ELEVENLABS_API_KEY='...'    (https://elevenlabs.io)")

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
<title>Tavily Maxer… Intelligence for Portfolio Managers</title>
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
  .tag { font-size: 18px; font-weight: 600; color: var(--text); margin: 0 0 10px; letter-spacing: -0.01em; }
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
  button.mic {
    align-self: flex-end; background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
    font-size: 16px; line-height: 1; cursor: pointer; transition: border-color .15s, background .15s;
  }
  button.mic:hover { border-color: var(--accent); }
  button.mic:disabled { opacity: .4; cursor: default; }
  button.mic.recording { border-color: var(--bad); background: rgba(255,107,107,.12); animation: micpulse 1.1s ease-in-out infinite; }
  @keyframes micpulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,107,107,.5); } 50% { box-shadow: 0 0 0 6px rgba(255,107,107,0); } }
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
  .answer table.md { border-collapse: collapse; margin: 10px 0 14px; font-size: 14px; display: block; overflow-x: auto; max-width: 100%; }
  .answer table.md th, .answer table.md td { border: 1px solid var(--border); padding: 7px 11px; text-align: left; vertical-align: top; }
  .answer table.md th { background: var(--panel-2); color: var(--text); font-weight: 600; white-space: nowrap; }
  .answer hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
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

  /* mode toggle */
  .modes { display: flex; justify-content: center; margin: 18px 0 4px; }
  .seg { display: inline-flex; background: var(--panel-2); border: 1px solid var(--border); border-radius: 999px; padding: 4px; gap: 4px; }
  .seg button { background: transparent; color: var(--muted); border: none; border-radius: 999px; padding: 7px 18px; font: inherit; font-size: 14px; cursor: pointer; }
  .seg button.active { background: linear-gradient(90deg, var(--accent), #6a5cff); color: #fff; font-weight: 600; }

  /* dropzone */
  .dropzone { margin-top: 14px; border: 1.5px dashed var(--border); border-radius: 14px; padding: 18px; text-align: center; color: var(--muted); cursor: pointer; transition: border-color .15s, background .15s; }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: rgba(79,124,255,.06); }
  .dropzone b { color: var(--text); }
  .dropzone .file { margin-top: 8px; color: var(--accent-2); font-size: 13px; }
  .hidden { display: none !important; }

  /* metrics */
  table.metrics { width: 100%; border-collapse: collapse; font-size: 14.5px; }
  table.metrics td { padding: 9px 8px; border-top: 1px solid var(--border); vertical-align: top; }
  table.metrics tr:first-child td { border-top: none; }
  table.metrics .mid { color: var(--accent-2); font-weight: 700; width: 30px; }
  table.metrics .mval { text-align: right; font-weight: 700; white-space: nowrap; }
  table.metrics .mdef { color: var(--muted); font-size: 12.5px; }
  .pill { display: inline-block; font-size: 13px; font-weight: 700; color: var(--accent-2); background: rgba(54,214,195,.12); border: 1px solid rgba(54,214,195,.3); border-radius: 6px; padding: 1px 6px; margin: 0 1px; cursor: pointer; }
  .pill.chartlink { color: var(--accent); background: rgba(79,124,255,.12); border-color: rgba(79,124,255,.3); }
  .chart { width: 100%; height: 360px; margin: 6px 0 18px; }
  .chart .cap { font-size: 12.5px; color: var(--muted); margin-bottom: 4px; }
  .chart.target { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 8px; }

  /* portfolio dashboard: compact overview on top, then charts + metrics side by side */
  .trio { display: block; }
  body.wide .wrap { max-width: 1280px; }
  .results.portfolio .trio {
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr);
    grid-template-areas: "overview overview" "charts metrics";
    gap: 18px; align-items: start;
  }
  .results.portfolio #overviewCard { grid-area: overview; }
  .results.portfolio #chartsCard { grid-area: charts; }
  .results.portfolio #metricsCard { grid-area: metrics; }
  /* min-width:0 lets grid cells shrink; overflow:hidden stops a chart from spilling over */
  .results.portfolio .trio .card { margin-bottom: 0; min-width: 0; overflow: hidden; }
  /* overview made smaller: full width but compact and scrollable */
  .results.portfolio #overviewCard .answer { font-size: 15px; line-height: 1.6; max-height: 300px; overflow-y: auto; }
  .results.portfolio #overviewCard .answer h1, .results.portfolio #overviewCard .answer h2, .results.portfolio #overviewCard .answer h3 { font-size: 16px; }
  .results.portfolio .chart { width: 100%; }
  /* headline metrics table at the top of the overview */
  table.highlights { width: 100%; border-collapse: collapse; margin-bottom: 14px; font-size: 14px; }
  table.highlights th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  table.highlights th:last-child, table.highlights .hv { text-align: right; }
  table.highlights td { padding: 8px; border-bottom: 1px solid var(--border); }
  table.highlights .hv { font-weight: 700; color: var(--accent-2); white-space: nowrap; }
  @media (max-width: 1024px) {
    .results.portfolio .trio { grid-template-columns: 1fr; grid-template-areas: "overview" "charts" "metrics"; }
    body.wide .wrap { max-width: 880px; }
  }
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="badge"><span class="dot"></span> Powered by Tavily web search · Nebius · LangChain</div>
      <h1>Tavily <span class="grad">Maxer</span></h1>
      <p class="tag">Intelligence for Portfolio Managers</p>
      <p class="sub" id="sub">Ask anything. Get a research answer where <b>every claim traces to a real
      retrieved source</b> — citations are validated, not improvised.</p>
    </header>

    <div class="modes">
      <div class="seg">
        <button id="modeResearch" class="active" type="button">Research</button>
        <button id="modePortfolio" type="button">Portfolio</button>
      </div>
    </div>

    <form id="form">
      <textarea id="q" placeholder="What changed in the AI search market this year?" rows="1"></textarea>
      <button class="mic" id="mic" type="button" title="Speak your question (ElevenLabs)" aria-label="Record voice question">&#127908;</button>
      <button class="ask" id="ask" type="submit">Search</button>
    </form>

    <div class="dropzone hidden" id="dropzone">
      <div><b>Drop a portfolio file</b> or click to choose — CSV, Excel, or a text list (ticker, weight).</div>
      <div class="file" id="fileName"></div>
      <input type="file" id="fileInput" class="hidden" accept=".csv,.xlsx,.xls,.txt" />
    </div>

    <div class="examples" id="examples">
      <span class="chip">What changed in the AI search market this year?</span>
      <span class="chip">Who won the most recent Nobel Prize in Physics?</span>
      <span class="chip">Compare the latest flagship phones from Apple and Google.</span>
      <span class="chip">What are the newest features in Python 3.13?</span>
    </div>

    <div class="status" id="status"><span class="spinner"></span><span id="statusText">Searching the web and grounding the answer…</span></div>

    <div class="results" id="results">
      <div class="trio">
        <div class="card" id="overviewCard">
          <h2 id="answerHead">Answer</h2>
          <div id="highlights"></div>
          <div class="answer" id="answer"></div>
        </div>
        <div class="card hidden" id="chartsCard">
          <h2>Charts</h2>
          <div id="charts"></div>
        </div>
        <div class="card hidden" id="metricsCard">
          <h2>Metrics</h2>
          <table class="metrics"><tbody id="metrics"></tbody></table>
        </div>
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
const micBtn = document.getElementById('mic');
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
const sub = document.getElementById('sub');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const chartsCard = document.getElementById('chartsCard');
const chartsEl = document.getElementById('charts');
const metricsCard = document.getElementById('metricsCard');
const metricsEl = document.getElementById('metrics');
const highlightsEl = document.getElementById('highlights');
const examplesEl = document.getElementById('examples');

// The headline metrics shown as a table at the top of the overview, in this order.
const HIGHLIGHT_KEYS = ['annualized_return', 'annualized_volatility', 'sharpe_ratio'];

let mode = 'research';            // 'research' | 'portfolio'
let uploadedFile = null;          // { name, b64 }
let metricMap = {};               // id -> metric record (for inline value chips)

const EXAMPLES = {
  research: [
    'What changed in the AI search market this year?',
    'Who won the most recent Nobel Prize in Physics?',
    'Compare the latest flagship phones from Apple and Google.',
    'What are the newest features in Python 3.13?',
  ],
  portfolio: [
    'Analyze a portfolio of 60% AAPL and 40% MSFT over the last 5 years.',
    'What is the 95% 1-day VaR of 50% SPY, 30% QQQ, 20% TLT?',
    'Compare the Sharpe ratio and drawdown of NVDA vs the S&P 500.',
  ],
};

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
    // [metric:k] -> a chip showing the computed value; [chart:j] -> a link to the chart.
    .replace(/\[metric:(\d+)\]/g, (_, id) => {
      const m = metricMap[id];
      return `<span class="pill" data-mid="${id}" title="${m ? escapeHtml(m.name) : ''}">${m ? escapeHtml(m.formatted) : 'metric ' + id}</span>`;
    })
    .replace(/\[chart:(\d+)\]/g, '<a class="pill chartlink" data-cid="$1">chart $1</a>')
    .replace(/\[(\d+)\]/g, '<a class="cite" href="#src-$1" data-id="$1">$1</a>');
  const closeList = () => { if (inList) { html += `</${inList}>`; inList = null; } };
  // A markdown table separator row: only spaces/pipes/colons/dashes, with at least one of each
  // structural char. e.g. |---|---|  or  :--- | ---:
  const isSep = (s) => { const t = s.trim(); return /^[\s|:-]+$/.test(t) && t.includes('-') && t.includes('|'); };
  const cells = (s) => s.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { closeList(); continue; }
    let m;
    // table: a row containing '|' immediately followed by a separator row
    if (line.includes('|') && i + 1 < lines.length && isSep(lines[i + 1])) {
      closeList();
      const head = cells(line);
      let body = '';
      i += 2;
      for (; i < lines.length && lines[i].trim() && lines[i].includes('|'); i++) {
        body += '<tr>' + cells(lines[i]).map(c => `<td>${inline(c)}</td>`).join('') + '</tr>';
      }
      i--;
      html += '<table class="md"><thead><tr>' + head.map(h => `<th>${inline(h)}</th>`).join('') +
              '</tr></thead><tbody>' + body + '</tbody></table>';
      continue;
    }
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) { closeList(); const n = Math.min(m[1].length, 3); html += `<h${n}>${inline(m[2])}</h${n}>`; }
    else if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) { closeList(); html += '<hr>'; }
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

// A small, code-computed headline table at the top of the overview (no LLM, no drift).
function renderHighlights(metrics) {
  const byKey = {};
  (metrics || []).forEach(m => { byKey[m.key] = m; });
  const rows = HIGHLIGHT_KEYS.map(k => byKey[k]).filter(Boolean);
  if (!rows.length) { highlightsEl.innerHTML = ''; return; }
  highlightsEl.innerHTML =
    '<table class="highlights"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>' +
    rows.map(m => `<tr><td>${escapeHtml(m.name)}</td><td class="hv">${escapeHtml(m.formatted)}</td></tr>`).join('') +
    '</tbody></table>';
}

function renderMetrics(metrics) {
  metricMap = {};
  (metrics || []).forEach(m => { metricMap[m.id] = m; });
  if (!metrics || !metrics.length) { metricsCard.classList.add('hidden'); return; }
  metricsCard.classList.remove('hidden');
  metricsEl.innerHTML = metrics.map(m => `
    <tr id="metric-${m.id}">
      <td class="mid">${m.id}</td>
      <td>${escapeHtml(m.name)}<div class="mdef">${escapeHtml(m.definition || '')}</div></td>
      <td class="mval">${escapeHtml(m.formatted)}</td>
    </tr>`).join('');
}

function renderCharts(charts) {
  if (!charts || !charts.length) { chartsCard.classList.add('hidden'); chartsEl.innerHTML = ''; return; }
  chartsCard.classList.remove('hidden');
  chartsEl.innerHTML = charts.map(c => `
    <div class="cap">[chart:${c.id}] ${escapeHtml(c.title)}</div>
    <div class="chart" id="chart-${c.id}"></div>`).join('');
  charts.forEach(c => {
    const spec = c.spec || {};
    try {
      Plotly.newPlot('chart-' + c.id, spec.data || [], Object.assign({ autosize: true }, spec.layout || {}),
                     { responsive: true, displayModeBar: false });
    } catch (e) {
      document.getElementById('chart-' + c.id).textContent = 'Could not render chart.';
    }
  });
}

// Plotly measures the container at plot time; since charts are first drawn while #results
// is still display:none, they come out at a default width and overflow. Re-fit them once the
// panel is visible and the grid has settled (and on any window resize).
function resizeCharts() {
  document.querySelectorAll('.chart').forEach(d => { try { Plotly.Plots.resize(d); } catch (e) {} });
}
window.addEventListener('resize', resizeCharts);

// Clicking a citation chip scrolls to its source; a chart link to its chart; a metric chip
// to its row.
answerEl.addEventListener('click', (e) => {
  const cite = e.target.closest('.cite');
  const chartLink = e.target.closest('.chartlink');
  const metricPill = e.target.closest('.pill:not(.chartlink)');
  const flash = (el) => { if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); el.classList.remove('target'); void el.offsetWidth; el.classList.add('target'); } };
  if (cite) { e.preventDefault(); flash(document.getElementById('src-' + cite.dataset.id)); }
  else if (chartLink) { e.preventDefault(); flash(document.getElementById('chart-' + chartLink.dataset.cid)); }
  else if (metricPill && metricPill.dataset.mid) { flash(document.getElementById('metric-' + metricPill.dataset.mid)); }
});

// ----- mode toggle -----
function setMode(next) {
  mode = next;
  const portfolio = mode === 'portfolio';
  document.getElementById('modeResearch').classList.toggle('active', !portfolio);
  document.getElementById('modePortfolio').classList.toggle('active', portfolio);
  dropzone.classList.toggle('hidden', !portfolio);
  askBtn.textContent = portfolio ? 'Analyze' : 'Search';
  q.placeholder = portfolio
    ? 'Analyze my portfolio… or: 60% AAPL, 40% MSFT over 5 years'
    : 'What changed in the AI search market this year?';
  sub.innerHTML = portfolio
    ? 'Upload a portfolio (or name tickers) and get <b>code-computed</b> risk/return metrics and interactive charts — every number is validated, never estimated by the model.'
    : 'Ask anything. Get a research answer where <b>every claim traces to a real retrieved source</b> — citations are validated, not improvised.';
  examplesEl.innerHTML = EXAMPLES[mode].map(x => `<span class="chip">${escapeHtml(x)}</span>`).join('');
}
document.getElementById('modeResearch').addEventListener('click', () => setMode('research'));
document.getElementById('modePortfolio').addEventListener('click', () => setMode('portfolio'));

// ----- file upload -----
dropzone.addEventListener('click', () => fileInput.click());
['dragover', 'dragenter'].forEach(ev => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.remove('drag')));
dropzone.addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer.files.length) readFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files.length) readFile(fileInput.files[0]); });

function readFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const b64 = String(reader.result).split(',')[1] || '';
    uploadedFile = { name: file.name, b64 };
    fileName.textContent = '✓ ' + file.name + ' — will be analyzed on Analyze';
  };
  reader.readAsDataURL(file);
}

async function ask() {
  const question = q.value.trim();
  const portfolio = mode === 'portfolio';
  if (!question && !(portfolio && uploadedFile)) return;

  errorBox.classList.remove('show');
  results.classList.remove('show');
  statusEl.classList.add('show');
  askBtn.disabled = true;
  statusText.textContent = portfolio
    ? 'Fetching prices and computing risk metrics…'
    : 'Searching the web and grounding the answer…';

  const payload = { question, portfolio_mode: portfolio };
  if (portfolio && uploadedFile) { payload.file_b64 = uploadedFile.b64; payload.filename = uploadedFile.name; }

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));

    renderMetrics(data.metrics);                 // sets metricMap, used by inline chips
    renderHighlights(data.metrics);              // headline table at top of overview
    answerEl.innerHTML = renderMarkdown(data.answer);
    renderCharts(data.charts);
    renderSources(data.sources, data.cited_source_ids);

    // Switch to the wide 3-column dashboard layout for portfolio results.
    const isPortfolioResult = !!((data.metrics && data.metrics.length) || (data.charts && data.charts.length));
    results.classList.toggle('portfolio', isPortfolioResult);
    document.body.classList.toggle('wide', isPortfolioResult);
    document.getElementById('answerHead').textContent = isPortfolioResult ? 'Overview' : 'Answer';

    if (data.validation.valid) {
      vbadge.className = 'vbadge ok';
      vbadge.innerHTML = portfolio
        ? '&#10003; Every metric and chart is code-computed and validated'
        : '&#10003; All citations verified against retrieved sources';
    } else {
      vbadge.className = 'vbadge fail';
      vbadge.innerHTML = '&#10007; ' + escapeHtml((data.validation.errors || []).join('; ') || 'Validation failed');
    }
    const parts = [];
    if (data.metrics && data.metrics.length) parts.push(`<b>${data.metrics.length}</b> metrics`);
    if (data.charts && data.charts.length) parts.push(`<b>${data.charts.length}</b> charts`);
    if (data.sources && data.sources.length) parts.push(`<b>${data.sources.length}</b> sources`);
    parts.push(`<b>${data.latency_seconds}s</b>`);
    parts.push(escapeHtml(data.model));
    stats.innerHTML = parts.join(' · ');

    statusEl.classList.remove('show');
    results.classList.add('show');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Now that the panel is visible and the grid is laid out, fit charts to their columns.
    requestAnimationFrame(resizeCharts);
  } catch (err) {
    statusEl.classList.remove('show');
    errorText.textContent = err.message;
    errorBox.classList.add('show');
  } finally {
    askBtn.disabled = false;
  }
}

// ----- voice input (ElevenLabs speech-to-text) -----
// Record from the mic with MediaRecorder, POST the raw audio to /api/transcribe (the
// server holds the ElevenLabs key), then drop the transcript into the textarea for the
// user to review and Search. Needs a secure context (localhost or HTTPS).
let mediaRecorder = null;
let audioChunks = [];

if (!navigator.mediaDevices || !window.MediaRecorder) {
  micBtn.disabled = true;
  micBtn.title = 'Voice input needs a secure (HTTPS/localhost) context and a supported browser.';
}

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    errorText.textContent = 'Microphone access was denied or unavailable.';
    errorBox.classList.add('show');
    return;
  }
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.addEventListener('dataavailable', (e) => { if (e.data.size) audioChunks.push(e.data); });
  mediaRecorder.addEventListener('stop', async () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
    await transcribe(blob);
  });
  mediaRecorder.start();
  micBtn.classList.add('recording');
  micBtn.title = 'Stop and transcribe';
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  micBtn.classList.remove('recording');
}

async function transcribe(blob) {
  errorBox.classList.remove('show');
  micBtn.disabled = true;
  const prevTitle = micBtn.title;
  micBtn.title = 'Transcribing…';
  try {
    const res = await fetch('/api/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': blob.type || 'audio/webm' },
      body: blob,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    const text = (data.text || '').trim();
    if (text) {
      q.value = q.value.trim() ? (q.value.trim() + ' ' + text) : text;
      q.dispatchEvent(new Event('input'));
      q.focus();
    } else {
      errorText.textContent = 'No speech was detected. Try again.';
      errorBox.classList.add('show');
    }
  } catch (err) {
    errorText.textContent = err.message;
    errorBox.classList.add('show');
  } finally {
    micBtn.disabled = false;
    micBtn.title = 'Speak your question (ElevenLabs)';
  }
}

micBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
});

form.addEventListener('submit', (e) => { e.preventDefault(); ask(); });
setMode('research');

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
