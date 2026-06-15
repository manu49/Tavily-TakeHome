"""`grounded` — ask a question, get an answer you can audit.

    uv run grounded "What changed in the AI search market this year?"
    uv run grounded --json "..."        # emit the full evidence ledger as JSON
    uv run grounded --no-verify "..."   # deterministic quote-check only (no judge)
    uv run grounded --baseline "..."    # naive starter-style answer (for contrast)
"""

from __future__ import annotations

from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Settings
from .observability import setup_tracing
from .pipeline import baseline_research, research
from .schemas import Verdict

app = typer.Typer(add_completion=False, help="A verifiable research agent built on Tavily.")
console = Console()

_VERDICT_STYLE = {
    Verdict.supported: "green",
    Verdict.partial: "yellow",
    Verdict.unsupported: "red",
}


@app.command()
def main(
    question: Annotated[list[str], typer.Argument(help="Your research question.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit the evidence ledger as JSON.")] = False,
    verify: Annotated[bool, typer.Option("--verify/--no-verify", help="Run the LLM verification gate.")] = True,
    baseline: Annotated[bool, typer.Option("--baseline", help="Run the naive starter-style agent instead.")] = False,
) -> None:
    load_dotenv()
    settings = Settings.from_env()

    missing = settings.missing_credentials()
    if missing:
        console.print(f"[bold red]Missing environment variables:[/bold red] {', '.join(missing)}")
        console.print("Add them to a .env file (see .env.example).")
        raise typer.Exit(code=1)

    question_text = " ".join(question)
    destination = setup_tracing(settings)

    console.print(Panel.fit(question_text, title="Question", border_style="cyan"))

    try:
        if baseline:
            answer, sources = baseline_research(settings, question_text)
            console.print(Panel(answer, title="Answer (baseline)", border_style="white"))
            _render_sources(sources)
        else:
            ledger = research(settings, question_text, verify=verify)
            if json_out:
                console.print_json(ledger.model_dump_json(indent=2))
            else:
                _render_ledger(ledger)
    except Exception as exc:  # surface a clean message, not a traceback
        console.print(f"[bold red]Run failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    if destination and destination != "console":
        console.print(f"\n[dim]Traces → {destination} (run `uv run phoenix serve` to view)[/dim]")


def _grounded_color(score: float) -> str:
    return "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"


def _render_ledger(ledger) -> None:
    color = _grounded_color(ledger.groundedness)
    console.print(
        Panel(
            ledger.answer,
            title=f"Answer  ·  groundedness [{color}]{ledger.groundedness:.0%}[/{color}]",
            border_style=color,
        )
    )

    if ledger.verifications:
        table = Table(title="Evidence ledger", show_lines=False, expand=True)
        table.add_column("Claim", ratio=3)
        table.add_column("Verdict", justify="center")
        table.add_column("Cites", justify="center")
        table.add_column("Why", ratio=2)
        for v in ledger.verifications:
            style = _VERDICT_STYLE.get(v.verdict, "white")
            cites = ", ".join(c.source_id for c in v.claim.citations) or "—"
            table.add_row(
                v.claim.text,
                f"[{style}]{v.verdict.value}[/{style}]",
                cites,
                v.reason,
            )
        console.print(table)

    _render_sources(ledger.sources)


def _render_sources(sources) -> None:
    if not sources:
        return
    table = Table(title="Sources", show_header=True, expand=True)
    table.add_column("#", justify="right")
    table.add_column("Title", ratio=2)
    table.add_column("URL", ratio=3)
    for s in sources:
        table.add_row(s.id, s.title or "—", s.url)
    console.print(table)


if __name__ == "__main__":
    app()
