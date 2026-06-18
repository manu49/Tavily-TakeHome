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

The page itself lives in `index.html`, which is shared with the Vercel deployment (see
`api/ask.py` + `vercel.json`). This server and the serverless function both POST to the
same `/api/ask` endpoint, so the same front end works in both places.

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
from pathlib import Path

from dotenv import load_dotenv

import tavily_maxer as tm

load_dotenv()
load_dotenv(".env.local", override=False)

INDEX_PATH = Path(__file__).resolve().parent / "index.html"


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
    return INDEX_PATH.read_bytes()


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


if __name__ == "__main__":
    main()
