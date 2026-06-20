"""Run tavily_maxer against a fixed set of sample questions and record I/O as a dataset.

Hits real Tavily + Nebius APIs (uses run_query, which also appends to logs/runs.jsonl
and -- if LANGSMITH_TRACING=true -- attaches citation-validity feedback in LangSmith).
Writes two artifacts, both meant to be committed as the run record:

  evals/dataset.csv  -- one row per question, tabular, easy to diff/load into pandas
  evals/dataset.md   -- the same data as a human-readable markdown report

Usage (from repo root):
  uv run evals/run_samples.py
  uv run evals/run_samples.py --questions evals/questions.json --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # entrypoints
sys.path.insert(0, str(ROOT / "lib"))    # supporting modules

import tavily_maxer as tm  # noqa: E402

CSV_FIELDS = [
    "id",
    "category",
    "question",
    "status",
    "answer",
    "cited_source_ids",
    "num_sources",
    "sources",
    "validation_valid",
    "validation_errors",
    "tool_call_count",
    "latency_seconds",
    "run_id",
    "error",
]


@dataclass
class SampleRow:
    id: str
    category: str
    question: str
    status: str
    answer: str = ""
    cited_source_ids: str = ""
    num_sources: int = 0
    sources: str = ""
    validation_valid: str = ""
    validation_errors: str = ""
    tool_call_count: int = 0
    latency_seconds: float = 0.0
    run_id: str = ""
    error: str = ""

    def to_csv_dict(self) -> dict:
        return {field: getattr(self, field) for field in CSV_FIELDS}


def run_one(item: dict, model: str) -> SampleRow:
    question = item["question"]
    console_label = f"[{item['id']}] {question}"
    print(f"Running {console_label} ...", flush=True)
    try:
        result = tm.run_query(question, model=model, log=True)
    except Exception as exc:  # noqa: BLE001 - record any failure mode for the dataset
        print(f"  FAILED: {exc}")
        return SampleRow(
            id=item["id"], category=item["category"], question=question,
            status="error", error=str(exc),
        )

    sources = "; ".join(
        f"[{r.id}] {r.title} ({r.url})" for r in result.registry.all_sources()
    )
    print(
        f"  ok in {result.latency_seconds:.1f}s, "
        f"{len(result.registry)} sources, "
        f"validation={'pass' if result.validation.valid else 'FAIL'}"
    )
    return SampleRow(
        id=item["id"],
        category=item["category"],
        question=question,
        status="ok",
        answer=result.answer.answer,
        cited_source_ids=json.dumps(sorted(set(result.answer.cited_source_ids))),
        num_sources=len(result.registry),
        sources=sources,
        validation_valid=str(result.validation.valid),
        validation_errors="; ".join(result.validation.errors),
        tool_call_count=len(result.tool_calls),
        latency_seconds=round(result.latency_seconds, 3),
        run_id=result.run_id or "",
    )


def write_csv(rows: list[SampleRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_markdown(rows: list[SampleRow], path: Path, *, model: str) -> None:
    ok_rows = [r for r in rows if r.status == "ok"]
    valid_rows = [r for r in ok_rows if r.validation_valid == "True"]
    pass_rate = f"{len(valid_rows)}/{len(ok_rows)}" if ok_rows else "0/0"
    avg_latency = (
        sum(r.latency_seconds for r in ok_rows) / len(ok_rows) if ok_rows else 0.0
    )

    lines = [
        "# tavily_maxer sample run dataset",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Model: {model}",
        f"- Runs: {len(rows)} ({len(ok_rows)} completed, {len(rows) - len(ok_rows)} failed to converge)",
        f"- Citation validation pass rate: {pass_rate}",
        f"- Average latency (completed runs): {avg_latency:.2f}s",
        "",
        "| id | category | question | status | sources | validation | latency (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        question = row.question.replace("|", "\\|")
        validation = row.validation_valid if row.status == "ok" else "n/a"
        lines.append(
            f"| {row.id} | {row.category} | {question} | {row.status} | "
            f"{row.num_sources} | {validation} | {row.latency_seconds or ''} |"
        )

    lines.append("")
    lines.append("## Full answers")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.id} -- {row.question}")
        lines.append("")
        if row.status == "ok":
            lines.append(row.answer)
            lines.append("")
            lines.append(f"**Sources:** {row.sources or '_none_'}")
            lines.append("")
            lines.append(
                f"**Validation:** {'pass' if row.validation_valid == 'True' else 'FAIL'}"
                + (f" -- {row.validation_errors}" if row.validation_errors else "")
            )
        else:
            lines.append(f"_Run failed: {row.error}_")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "evals" / "questions.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals")
    parser.add_argument("--model", default=tm.DEFAULT_MODEL)
    args = parser.parse_args()

    for name in ("TAVILY_API_KEY", "NEBIUS_API_KEY"):
        if not __import__("os").getenv(name):
            raise SystemExit(f"Missing {name}; set it in the environment or .env/.env.local")

    questions = json.loads(args.questions.read_text())

    rows: list[SampleRow] = []
    start = time.perf_counter()
    for item in questions:
        rows.append(run_one(item, args.model))
    total = time.perf_counter() - start

    write_csv(rows, args.output_dir / "dataset.csv")
    write_markdown(rows, args.output_dir / "dataset.md", model=args.model)

    ok = sum(1 for r in rows if r.status == "ok")
    valid = sum(1 for r in rows if r.validation_valid == "True")
    print(
        f"\nDone in {total:.1f}s: {ok}/{len(rows)} runs completed, "
        f"{valid}/{ok or 1} passed citation validation."
    )
    print(f"Wrote {args.output_dir / 'dataset.csv'} and {args.output_dir / 'dataset.md'}")


if __name__ == "__main__":
    main()
