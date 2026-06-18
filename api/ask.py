"""
api/ask.py — Vercel Python serverless function backing the demo page.

Vercel maps this file to the route `/api/ask`, which is exactly what `index.html` POSTs
to (the same endpoint the local `webapp.py` dev server serves). The function runs the
real tavily_maxer agent and returns the shaped JSON result.

Differences from the local dev server, because of the serverless environment:
  - `log=False`: the function filesystem is read-only except /tmp, so we skip writing
    logs/runs.jsonl. LangSmith tracing (if LANGSMITH_TRACING=true is set in the Vercel
    env) still auto-instruments the run via env vars — that path is unaffected.
  - `repair_attempts=1`: each citation-repair pass is another full agent turn; one keeps
    us comfortably inside the function's maxDuration (see vercel.json).

Required env vars (set in the Vercel dashboard → Project → Settings → Environment
Variables): TAVILY_API_KEY, NEBIUS_API_KEY. Optional: LANGSMITH_TRACING, LANGSMITH_API_KEY,
LANGSMITH_PROJECT.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

# The function's working dir is api/; add the repo root so `import tavily_maxer` resolves.
# tavily_maxer.py is bundled via "includeFiles" in vercel.json.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tavily_maxer as tm  # noqa: E402


def answer_question(question: str) -> dict:
    result = tm.run_query(question, model=tm.DEFAULT_MODEL, log=False, repair_attempts=1)
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


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel requires the name `handler`
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
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
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:  # noqa: N802 - a friendly hint if visited directly
        self._send_json(405, {"error": "POST a JSON body {\"question\": \"...\"} to this endpoint."})
