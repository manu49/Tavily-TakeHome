"""The research pipeline: plan -> retrieve -> synthesize -> verify.

This is a deliberately *orchestrated* pipeline rather than a free-form agent
tool-loop. For a system whose whole point is auditable, reproducible answers,
determinism is a feature: each step is a discrete, traced unit, and the same
question yields a comparable trace every time. A `create_agent` loop with
verification middleware would be the natural next step if we wanted the model
to decide its own retrieval trajectory.
"""

from __future__ import annotations

from opentelemetry import trace

from .config import Settings, build_chat_model
from .schemas import DraftAnswer, EvidenceLedger, SearchPlan, Source
from .retrieval import gather_sources
from .verify import EntailmentJudgment, assemble_ledger, verify_claims

tracer = trace.get_tracer("grounded.pipeline")

# Context-engineering knob: how much of each extracted source to put in the
# synthesis prompt. Full pages blow the budget; this keeps the most relevant
# head of each source while staying well within context.
_SOURCE_CHAR_BUDGET = 6000

_PLANNER_SYSTEM = (
    "You plan web research. Decompose the user's question into focused, "
    "non-overlapping search sub-queries. For each, choose the best Tavily "
    "topic: 'news' for current events, 'finance' for markets/companies/"
    "tickers, otherwise 'general'. Set time_range (day/week/month/year) only "
    "when recency genuinely matters. Favor diverse angles over near-duplicates."
)

_SYNTH_SYSTEM = (
    "You are a research analyst. Answer the question using ONLY the provided "
    "sources. Decompose your answer into atomic claims. Every claim MUST cite "
    "the source id(s) it relies on (e.g. S2) and include a SHORT VERBATIM "
    "quote copied exactly from that source's text — do not paraphrase the "
    "quote. Never assert anything you cannot quote. Prefer fewer, well-"
    "supported claims over broad coverage. In `summary`, write the prose "
    "answer with inline [S1]/[S2] markers. If the sources do not answer the "
    "question, say so plainly."
)


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n].rstrip() + " …[truncated]"


def plan_searches(settings: Settings, question: str) -> SearchPlan:
    with tracer.start_as_current_span("pipeline.plan") as span:
        model = build_chat_model(settings, max_tokens=1024).with_structured_output(
            SearchPlan
        )
        plan: SearchPlan = model.invoke(
            [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
        plan.subqueries = plan.subqueries[: settings.max_subqueries]
        span.set_attribute("plan.subquery_count", len(plan.subqueries))
        return plan


def _sources_block(sources: list[Source]) -> str:
    parts = []
    for s in sources:
        parts.append(
            f"[{s.id}] {s.title}\nURL: {s.url}\n{_clip(s.content, _SOURCE_CHAR_BUDGET)}"
        )
    return "\n\n---\n\n".join(parts)


def synthesize(settings: Settings, question: str, sources: list[Source]) -> DraftAnswer:
    with tracer.start_as_current_span("pipeline.synthesize") as span:
        model = build_chat_model(settings, max_tokens=4096).with_structured_output(
            DraftAnswer
        )
        human = (
            f"QUESTION:\n{question}\n\nSOURCES:\n{_sources_block(sources)}"
        )
        draft: DraftAnswer = model.invoke(
            [
                {"role": "system", "content": _SYNTH_SYSTEM},
                {"role": "user", "content": human},
            ]
        )
        span.set_attribute("synthesize.claim_count", len(draft.claims))
        return draft


def research(
    settings: Settings, question: str, *, verify: bool = True
) -> EvidenceLedger:
    """Run the full verifiable-research pipeline and return an EvidenceLedger."""
    with tracer.start_as_current_span("grounded.research") as span:
        span.set_attribute("question", question)

        plan = plan_searches(settings, question)
        sources = gather_sources(plan.subqueries, settings)

        if not sources:
            return EvidenceLedger(
                question=question,
                answer="I could not retrieve any sources to ground an answer.",
                sources=[],
                verifications=[],
                groundedness=0.0,
            )

        draft = synthesize(settings, question, sources)

        judge = None
        if verify:
            judge = build_chat_model(
                settings, for_judge=True, max_tokens=1024
            ).with_structured_output(EntailmentJudgment)

        verifications = verify_claims(draft, sources, judge=judge)
        ledger = assemble_ledger(
            question, draft, sources, verifications, settings.min_groundedness
        )
        span.set_attribute("grounded.groundedness", ledger.groundedness)
        span.set_attribute("grounded.flagged_count", len(ledger.flagged_claims))
        return ledger


# --------------------------------------------------------------------------- #
# Baseline: a faithful stand-in for the starter agent, for A/B evaluation.
# Single search, snippets only, free-form answer that is merely *asked* to cite.
# (The assignment forbids shipping starter_agent.py, so we reproduce its
#  behaviour here purely as the eval's control arm.)
# --------------------------------------------------------------------------- #
_BASELINE_SYSTEM = (
    "You are a concise research assistant. Use the search results to answer "
    "the question directly and include source URLs when available."
)


def baseline_research(
    settings: Settings, question: str
) -> tuple[str, list[Source]]:
    from langchain_tavily import TavilySearch

    with tracer.start_as_current_span("baseline.research") as span:
        payload = TavilySearch(max_results=settings.results_per_query).invoke(
            {"query": question}
        )
        results = payload.get("results", []) if isinstance(payload, dict) else []
        sources = [
            Source(
                id=f"S{i}",
                url=r.get("url", ""),
                title=r.get("title", "") or "",
                content=" ".join((r.get("content") or "").split()),  # snippet only
                score=float(r.get("score") or 0.0),
            )
            for i, r in enumerate(results, start=1)
            if r.get("url")
        ]
        block = "\n\n".join(f"[{s.id}] {s.title} ({s.url})\n{s.content}" for s in sources)
        model = build_chat_model(settings, max_tokens=2048)
        msg = model.invoke(
            [
                {"role": "system", "content": _BASELINE_SYSTEM},
                {"role": "user", "content": f"{question}\n\nSEARCH RESULTS:\n{block}"},
            ]
        )
        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
        span.set_attribute("baseline.source_count", len(sources))
        return answer, sources
