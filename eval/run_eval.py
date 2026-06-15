"""Offline evaluation: grounded pipeline vs. naive baseline.

For every question we run both systems, then score each answer with the same
independent judge (eval/judge.py) against the sources that system retrieved.
The headline metric is *groundedness* — the share of an answer's claims that
its own sources actually support — which is exactly the faithfulness property
the grounded pipeline is built to improve.

    uv run python eval/run_eval.py            # full set
    uv run python eval/run_eval.py --limit 3  # quick smoke
    uv run python eval/run_eval.py --no-verify
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).parent))  # allow `import judge`

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from grounded.config import Settings  # noqa: E402
from grounded.observability import setup_tracing  # noqa: E402
from grounded.pipeline import baseline_research, research  # noqa: E402
from judge import judge_answer  # noqa: E402

console = Console()
HERE = pathlib.Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
RESULTS_DIR = HERE / "results"


def load_dataset(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _safe_mean(xs: list[float]) -> float:
    return round(mean(xs), 3) if xs else 0.0


def run(limit: int | None, verify: bool) -> None:
    load_dotenv()
    settings = Settings.from_env()
    missing = settings.missing_credentials()
    if missing:
        console.print(f"[bold red]Missing:[/bold red] {', '.join(missing)} — see .env.example")
        raise SystemExit(1)

    setup_tracing(settings)
    dataset = load_dataset(limit)
    console.print(f"Evaluating [bold]{len(dataset)}[/bold] questions on model [cyan]{settings.model}[/cyan]\n")

    rows = []
    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        console.print(f"[dim]{i}/{len(dataset)}[/dim] {q}")

        ledger = research(settings, q, verify=verify)
        asserted = ledger.answer.split("⚠ Unverified")[0].strip()  # judge what it asserts
        g_eval = judge_answer(settings, q, asserted, ledger.sources)

        b_answer, b_sources = baseline_research(settings, q)
        b_eval = judge_answer(settings, q, b_answer, b_sources)

        hallucinated = sum(1 for v in ledger.verifications if not v.quote_found)
        rows.append(
            {
                "id": item["id"],
                "question": q,
                "grounded": {
                    "external_groundedness": round(g_eval.groundedness, 3),
                    "internal_groundedness": ledger.groundedness,
                    "claims": g_eval.total_claims,
                    "sources": len(ledger.sources),
                    "flagged": len(ledger.flagged_claims),
                    "hallucinated_quotes": hallucinated,
                    "citation_quality": round(g_eval.citation_quality, 3),
                },
                "baseline": {
                    "external_groundedness": round(b_eval.groundedness, 3),
                    "claims": b_eval.total_claims,
                    "sources": len(b_sources),
                    "citation_quality": round(b_eval.citation_quality, 3),
                },
            }
        )

    _report(rows, settings, verify)


def _report(rows: list[dict], settings: Settings, verify: bool) -> None:
    g = [r["grounded"] for r in rows]
    b = [r["baseline"] for r in rows]

    summary = {
        "model": settings.model,
        "judge_model": settings.judge_model,
        "verify": verify,
        "n": len(rows),
        "grounded": {
            "groundedness": _safe_mean([x["external_groundedness"] for x in g]),
            "citation_quality": _safe_mean([x["citation_quality"] for x in g]),
            "avg_sources": _safe_mean([x["sources"] for x in g]),
            "avg_flagged": _safe_mean([x["flagged"] for x in g]),
            "hallucinated_quotes": sum(x["hallucinated_quotes"] for x in g),
        },
        "baseline": {
            "groundedness": _safe_mean([x["external_groundedness"] for x in b]),
            "citation_quality": _safe_mean([x["citation_quality"] for x in b]),
            "avg_sources": _safe_mean([x["sources"] for x in b]),
        },
    }

    table = Table(title="Grounded vs. baseline", expand=True)
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Grounded", justify="right")
    table.add_column("Δ", justify="right")

    def add(metric, bv, gv, pct=True):
        delta = gv - bv
        fmt = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.2f}")
        style = "green" if delta >= 0 else "red"
        table.add_row(metric, fmt(bv), fmt(gv), f"[{style}]{'+' if delta >= 0 else ''}{fmt(delta)}[/{style}]")

    add("Groundedness", summary["baseline"]["groundedness"], summary["grounded"]["groundedness"])
    add("Citation quality", summary["baseline"]["citation_quality"], summary["grounded"]["citation_quality"])
    add("Avg sources used", summary["baseline"]["avg_sources"], summary["grounded"]["avg_sources"], pct=False)
    console.print()
    console.print(table)
    console.print(
        f"Grounded also caveated [yellow]{summary['grounded']['avg_flagged']}[/yellow] claims/question "
        f"on average and caught [red]{summary['grounded']['hallucinated_quotes']}[/red] fabricated quotes."
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"eval-{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    console.print(f"\n[dim]Wrote {out}[/dim]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    args = ap.parse_args()
    run(args.limit, args.verify)
