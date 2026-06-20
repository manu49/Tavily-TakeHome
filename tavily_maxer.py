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
tavily_maxer: Tavily + LangChain research agent with grounded citations and observability.

Built on top of starter_agent.py. Adds three things the starter agent doesn't have:

  1. Structurally enforced citations. A `SourceRegistry` wraps the Tavily tool, dedupes
     results by normalized URL (fixing a duplicate-result bug observed in the starter
     agent), and stamps every unique result with a stable, code-owned integer id. The
     model answers through a `response_format=ResearchAnswer` schema instead of free
     text, so citation markers refer to real ids instead of being improvised.
  2. Deterministic validation. After every run, `validate_citations` checks that every
     cited id (declared or inline `[n]`) actually exists in the registry. A passing
     check means zero hallucinated sources, by construction.
  3. Observability. Every run is appended to logs/runs.jsonl (question, tool calls,
     latency, validation outcome) so behavior is inspectable without any external
     service. If LANGSMITH_TRACING=true is set, the run is also traced in LangSmith and
     the validation result is attached as run feedback.

Setup is identical to starter_agent.py:
  1. Tavily API key: https://app.tavily.com
  2. Nebius API key: https://tokenfactory.nebius.com
  3. Export both or put them in a .env / .env.local file:
       TAVILY_API_KEY="tvly-..."
       NEBIUS_API_KEY="..."
  4. Run:
       uv run tavily_maxer.py "What changed in the AI search market this year?"

Optional observability:
       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY="..."
       LANGSMITH_PROJECT="tavily-maxer"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Supporting library modules (quant, charts, portfolio, portfolio_tool, ...) live in lib/
# to keep the repo root uncluttered. Put it on sys.path so they keep importing by name.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import typer
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tracers.context import collect_runs
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field, PrivateAttr
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

load_dotenv()
load_dotenv(".env.local", override=False)

DEFAULT_MODEL = "openai/gpt-oss-120b"
RUNS_LOG_PATH = Path("logs/runs.jsonl")
CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

console = Console()


# --------------------------------------------------------------------------------------
# Source registry: dedupe Tavily results and hand out stable, code-owned citation ids.
# --------------------------------------------------------------------------------------

_TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "mc_cid", "mc_eid",
}


def normalize_url(url: str) -> str:
    """Canonicalize a URL so equivalent links (http/https, www, trailing slash,
    tracking params, fragments) collapse to the same dedupe key."""
    parsed = urlsplit(url.strip())
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    query_pairs = sorted(
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_QUERY_KEYS
    )
    return urlunsplit(("https", netloc, path, urlencode(query_pairs), ""))


@dataclass
class SourceRecord:
    id: int
    title: str
    url: str
    normalized_url: str


class SourceRegistry:
    """Single source of truth for sources seen during one agent run.

    Assigns each unique URL a stable integer id the first time it's seen, regardless
    of which tool call (or how many duplicate entries within that call) it came from.
    """

    def __init__(self) -> None:
        self._by_normalized_url: Dict[str, SourceRecord] = {}
        self._by_id: Dict[int, SourceRecord] = {}
        self._next_id = 1

    def register(self, results: List[dict]) -> tuple[List[dict], int]:
        """Register raw Tavily result dicts.

        Returns (labeled_results, duplicates_removed). `labeled_results` has one entry
        per result passed in that had a URL, each tagged with its registry `id` and
        canonical `title`/`url`. `duplicates_removed` counts entries that pointed at a
        URL already registered, either earlier in this same call or in a prior call.
        """
        labeled: List[dict] = []
        duplicates = 0
        seen_this_call: set[str] = set()

        for result in results:
            url = (result.get("url") or "").strip()
            if not url:
                continue
            key = normalize_url(url)

            if key in seen_this_call:
                duplicates += 1
                continue
            seen_this_call.add(key)

            record = self._by_normalized_url.get(key)
            if record is None:
                record = SourceRecord(
                    id=self._next_id,
                    title=(result.get("title") or url).strip(),
                    url=url,
                    normalized_url=key,
                )
                self._by_normalized_url[key] = record
                self._by_id[record.id] = record
                self._next_id += 1
            else:
                duplicates += 1

            labeled.append({**result, "id": record.id, "title": record.title, "url": record.url})

        return labeled, duplicates

    def get(self, source_id: int) -> Optional[SourceRecord]:
        return self._by_id.get(source_id)

    def ids(self) -> set[int]:
        return set(self._by_id.keys())

    def all_sources(self) -> List[SourceRecord]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def to_markdown(self) -> str:
        if not self._by_id:
            return "_No sources retrieved._"
        lines = [f"[{r.id}] {r.title} — {r.url}" for r in self.all_sources()]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._by_id)


class TavilySearchWithRegistry(TavilySearch):
    """TavilySearch wrapped with a SourceRegistry.

    Intercepts raw results before they reach the model: dedupes by normalized URL
    (fixes duplicate-result entries seen in starter_agent testing) and stamps each
    unique result with a stable integer `id`, owned by code rather than the LLM, so
    the model has something concrete to cite.
    """

    _registry: SourceRegistry = PrivateAttr(default_factory=SourceRegistry)

    @property
    def registry(self) -> SourceRegistry:
        return self._registry

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced", "fast", "ultra-fast"]] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raw = super()._run(
            query=query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            search_depth=search_depth,
            include_images=include_images,
            time_range=time_range,
            topic=topic,
            start_date=start_date,
            end_date=end_date,
            run_manager=run_manager,
            **kwargs,
        )
        if not isinstance(raw, dict):
            return raw

        labeled, duplicates = self._registry.register(raw.get("results", []))
        return {
            **raw,
            "results": labeled,
            "duplicate_results_removed": duplicates,
            "note": (
                "Each result has a stable numeric 'id'. Cite sources in your final answer "
                "using inline [id] markers that match these ids exactly."
            ),
        }


# --------------------------------------------------------------------------------------
# Structured output schema
# --------------------------------------------------------------------------------------

class Source(BaseModel):
    id: int
    title: str
    url: str


class ResearchAnswer(BaseModel):
    """Final answer, forced through structured output instead of free text."""

    answer: str = Field(
        description=(
            "Markdown answer to the user's question with inline [n] citation markers, "
            "e.g. 'Paris is the capital of France [1].' Every non-trivial factual claim "
            "should carry a marker pointing at a source id returned by tavily_search."
        )
    )
    cited_source_ids: List[int] = Field(
        default_factory=list,
        description="Every source id referenced anywhere in `answer`, deduplicated.",
    )
    referenced_metric_ids: List[int] = Field(
        default_factory=list,
        description=(
            "Every metric id referenced in `answer` as [metric:k], deduplicated. Only used "
            "for quantitative/portfolio questions where the analyze_portfolio tool returned "
            "metrics; leave empty for pure web-research answers."
        ),
    )
    referenced_chart_ids: List[int] = Field(
        default_factory=list,
        description=(
            "Every chart id embedded in `answer` as [chart:j], deduplicated. Only used when "
            "analyze_portfolio returned charts; leave empty otherwise."
        ),
    )


SYSTEM_PROMPT = """You are a meticulous research assistant backed by Tavily web search.

Rules:
- Use the tavily_search tool for any question that depends on current or factual web
  information. Each result the tool returns is pre-labeled with a stable numeric "id".
- Every non-trivial factual claim in your final answer must carry an inline citation
  marker like [3] referencing one of those ids.
- Only ever cite ids that were actually returned by tavily_search in this conversation.
  Never invent an id, a marker, or a URL, and never cite a source you have not seen.
- Use ONLY the plain ASCII format [n] for citations, e.g. [1] or [2][5]. Never use any
  other citation style -- no `【n†source】`, no footnote symbols, no raw URLs inline.
- If no retrieved source supports a claim, state that plainly instead of forcing a
  citation onto it.
- Populate cited_source_ids with every id you cited in the answer text.
"""


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

# Citation-shaped markers the model is *not* supposed to use (e.g. OpenAI-harmony-style
# 【0†L8-L15】 footnotes seen in starter_agent testing). Catching these is the whole
# point of validation: a model that reverts to its pretrained citation habit produces
# markers that point at nothing, and a regex that only looks for well-formed `[n]`
# would silently pass that answer.
STRAY_MARKER_RE = re.compile(r"【[^】]*】")

# Metric references are syntactically distinct from source citations: `[metric:3]` vs `[3]`.
# CITATION_MARKER_RE (\[(\d+)\]) cannot match `[metric:3]`, so the two namespaces never
# collide -- a number can be both source id 3 and metric id 3 without ambiguity.
METRIC_MARKER_RE = re.compile(r"\[metric:(\d+)\]")
CHART_MARKER_RE = re.compile(r"\[chart:(\d+)\]")

# gpt-oss-120b frequently reverts to its pretrained full-width citation habit, emitting
# 【1】 instead of [1] (often several per answer). When the content between the brackets is
# just source-id number(s), that's a real citation wearing the wrong skin -- normalize it
# deterministically to [n] rather than relying on the (flaky) model to reformat on a repair
# turn. Markers with extra structure (e.g. 【1†L8-L15】) are left untouched so they still
# fail validation as genuinely malformed.
_FULLWIDTH_CITATION_RE = re.compile(r"【\s*(\d+(?:\s*,\s*\d+)*)\s*】")


def normalize_citation_markers(text: str) -> str:
    """Rewrite full-width numeric citation markers (【1】, 【1, 3】) to ASCII [1], [1][3]."""
    return _FULLWIDTH_CITATION_RE.sub(
        lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))), text
    )


@dataclass
class ValidationResult:
    valid: bool
    declared_ids: set[int] = field(default_factory=set)
    inline_ids: set[int] = field(default_factory=set)
    missing_ids: set[int] = field(default_factory=set)
    stray_markers: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    declared_metric_ids: set[int] = field(default_factory=set)
    inline_metric_ids: set[int] = field(default_factory=set)
    missing_metric_ids: set[int] = field(default_factory=set)
    declared_chart_ids: set[int] = field(default_factory=set)
    inline_chart_ids: set[int] = field(default_factory=set)
    missing_chart_ids: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "declared_ids": sorted(self.declared_ids),
            "inline_ids": sorted(self.inline_ids),
            "missing_ids": sorted(self.missing_ids),
            "stray_markers": self.stray_markers,
            "errors": self.errors,
            "declared_metric_ids": sorted(self.declared_metric_ids),
            "inline_metric_ids": sorted(self.inline_metric_ids),
            "missing_metric_ids": sorted(self.missing_metric_ids),
            "declared_chart_ids": sorted(self.declared_chart_ids),
            "inline_chart_ids": sorted(self.inline_chart_ids),
            "missing_chart_ids": sorted(self.missing_chart_ids),
        }


def validate_artifacts(
    answer: ResearchAnswer,
    source_registry: SourceRegistry,
    metric_registry: Optional[Any] = None,
    chart_registry: Optional[Any] = None,
) -> ValidationResult:
    """Deterministic post-hoc check over every code-owned artifact the answer references:

    1. Every cited source id (declared in `cited_source_ids` or inline `[n]`) must exist in
       the SourceRegistry -- a pass means zero hallucinated sources, by construction.
    2. Every referenced metric id (`referenced_metric_ids` or inline `[metric:k]`) must
       exist in the MetricRegistry -- zero fabricated numbers, by construction.
    3. Every referenced chart id (`referenced_chart_ids` or inline `[chart:j]`) must exist
       in the ChartRegistry -- the model can't embed a figure that was never built.
    When the matching registry is absent (e.g. a pure research run), any reference of that
    kind is treated as invalid.
    4. The answer must contain no improvised, non-`[n]` citation-shaped marker (`【1†...】`).
    """
    valid_ids = source_registry.ids()
    declared = set(answer.cited_source_ids)
    inline = {int(m) for m in CITATION_MARKER_RE.findall(answer.answer)}
    missing = (declared | inline) - valid_ids

    valid_metric_ids = metric_registry.ids() if metric_registry is not None else set()
    declared_metric = set(getattr(answer, "referenced_metric_ids", []) or [])
    inline_metric = {int(m) for m in METRIC_MARKER_RE.findall(answer.answer)}
    missing_metric = (declared_metric | inline_metric) - valid_metric_ids

    valid_chart_ids = chart_registry.ids() if chart_registry is not None else set()
    declared_chart = set(getattr(answer, "referenced_chart_ids", []) or [])
    inline_chart = {int(m) for m in CHART_MARKER_RE.findall(answer.answer)}
    missing_chart = (declared_chart | inline_chart) - valid_chart_ids

    stray_markers = STRAY_MARKER_RE.findall(answer.answer)

    errors: List[str] = []
    if missing:
        errors.append(f"Cited source id(s) not found in registry: {sorted(missing)}")
    if missing_metric:
        errors.append(f"Referenced metric id(s) not found in registry: {sorted(missing_metric)}")
    if missing_chart:
        errors.append(f"Referenced chart id(s) not found in registry: {sorted(missing_chart)}")
    if stray_markers:
        errors.append(f"Non-standard citation marker(s) found (expected [n]): {stray_markers}")

    return ValidationResult(
        valid=not errors,
        declared_ids=declared,
        inline_ids=inline,
        missing_ids=missing,
        stray_markers=stray_markers,
        errors=errors,
        declared_metric_ids=declared_metric,
        inline_metric_ids=inline_metric,
        missing_metric_ids=missing_metric,
        declared_chart_ids=declared_chart,
        inline_chart_ids=inline_chart,
        missing_chart_ids=missing_chart,
    )


def validate_citations(answer: ResearchAnswer, registry: SourceRegistry) -> ValidationResult:
    """Source-only validation (backward-compatible wrapper around validate_artifacts).

    Used by the research-only paths; equivalent to validate_artifacts with no metric
    registry.
    """
    return validate_artifacts(answer, registry, metric_registry=None)


# --------------------------------------------------------------------------------------
# Observability: local JSONL log + optional LangSmith feedback
# --------------------------------------------------------------------------------------

def log_run(record: dict, path: Path = RUNS_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def langsmith_tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}


def attach_langsmith_feedback(run_id: Optional[str], validation: ValidationResult) -> None:
    if not run_id or not langsmith_tracing_enabled():
        return
    try:
        from langsmith import Client

        Client().create_feedback(
            run_id,
            key="citation_validity",
            score=1 if validation.valid else 0,
            comment="; ".join(validation.errors) or "all citations resolve to registered sources",
        )
    except Exception as exc:  # pragma: no cover - best-effort, network/SDK dependent
        console.print(f"[dim yellow]LangSmith feedback skipped: {exc}[/dim yellow]")


# --------------------------------------------------------------------------------------
# Agent construction + core (non-streaming) run, used by both the CLI and evals/tests
# --------------------------------------------------------------------------------------

class FixedChatNebius(ChatNebius):
    """Works around a Nebius/vLLM streaming bug in gpt-oss models: the final
    argument fragment of a tool call is sometimes reported under a new
    `index` instead of the one it belongs to. LangChain merges tool_call
    chunks by index, so the stray fragment turns into a second, malformed
    tool call (no id/name) that the API rejects on the next turn. Chunks
    with no id/name are re-attached to the most recently started tool call
    instead of trusting the server-reported index.
    """

    _active_tool_call_index: Optional[int] = PrivateAttr(default=None)

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk

        tool_call_chunks = getattr(generation_chunk.message, "tool_call_chunks", None)
        if not tool_call_chunks:
            return generation_chunk

        for tool_call_chunk in tool_call_chunks:
            if tool_call_chunk.get("id"):
                self._active_tool_call_index = tool_call_chunk.get("index")
            elif self._active_tool_call_index is not None:
                tool_call_chunk["index"] = self._active_tool_call_index

        return generation_chunk


def parse_structured_fallback(messages: List[Any]) -> Optional[ResearchAnswer]:
    """Recover a structured answer when the model didn't go through the forced
    ResearchAnswer tool call.

    `create_agent` binds the final turn with `tool_choice="required"` so the model is
    supposed to call the structured-output tool. In testing, Nebius's gpt-oss-120b
    endpoint doesn't reliably honor that constraint: instead of a tool call, it
    sometimes answers with a plain-text AI message containing the exact same JSON
    shape `create_agent` would have parsed from a tool call. Rather than failing the
    run outright, treat that text as the structured payload if it parses cleanly.
    """
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip().startswith("{"):
            try:
                return ResearchAnswer.model_validate_json(content)
            except Exception:
                return None
        return None
    return None


_CODE_FENCE_RE = re.compile(r"\A```[a-zA-Z]*\n?|\n?```\Z")


def _message_text(message: Any) -> str:
    """Flatten an AI message's content (str, or a list of content blocks) to plain text."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def coerce_research_answer(messages: List[Any]) -> Optional[ResearchAnswer]:
    """Last-resort recovery when neither a structured response nor a clean JSON text
    payload (parse_structured_fallback) is available.

    gpt-oss-120b sometimes ignores the forced ResearchAnswer tool call entirely and just
    answers in free-text prose. Rather than 500 the whole request, salvage that final
    prose as the answer (lifting any inline [n] markers into cited_source_ids) so the user
    still gets a result that goes through the normal validation/repair path. Citations are
    still validated against the registry, so this can't smuggle in hallucinated sources --
    a bad marker just shows up as a failed badge instead of a crash.

    Kept separate from parse_structured_fallback so that function's strict, well-tested
    JSON-only contract is unchanged.
    """
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        text = _message_text(message).strip()
        if not text:
            continue  # skip tool-call-only AI turns; keep looking for the final prose
        unfenced = _CODE_FENCE_RE.sub("", text).strip() if text.startswith("```") else text
        if unfenced.startswith("{"):
            try:
                return ResearchAnswer.model_validate_json(unfenced)
            except Exception:
                pass
        cited = sorted({int(m) for m in CITATION_MARKER_RE.findall(text)})
        return ResearchAnswer(answer=text, cited_source_ids=cited)
    return None


PORTFOLIO_PROMPT_SUFFIX = """

If the user asks about "my portfolio" or "my holdings", first call `get_uploaded_portfolio`
(when available) to retrieve their tickers and weights, then pass those to
`analyze_portfolio`.

You also have an `analyze_portfolio` tool for quantitative portfolio questions (risk/return
metrics for a set of tickers and weights). When a question calls for it:
- Call `analyze_portfolio` instead of computing any numbers yourself. It returns metrics
  pre-labeled with stable ids like [metric:3]. You must NEVER compute or estimate a Sharpe
  ratio, volatility, VaR, or any other figure on your own -- only report ids it returned.
- Cite every quantitative claim with the exact [metric:k] id, e.g. "The portfolio's Sharpe
  ratio is 0.51 [metric:3]." Populate referenced_metric_ids with every id you cited.
- For a forward-looking question (e.g. "expected return next year"), do not assert a single
  fabricated forecast: present the historical metrics you computed AND, if useful, sourced
  analyst estimates via tavily_search [n], clearly labeled as estimates/assumptions.
- State the assumptions behind a metric (risk-free rate, VaR confidence and horizon) where
  relevant. This is analysis, not investment advice.
- Report the ACTUAL weights from get_uploaded_portfolio / analyze_portfolio. Never describe
  a portfolio as "equal-weight" or invent percentages -- use the weights you were given.
- Refer to each holding by its ticker symbol. Do NOT state a security's full name, asset
  class, or type (e.g. stock vs ETF) unless you confirmed it via tavily_search -- never
  guess what a ticker is.
"""


def build_agent(
    model: str = DEFAULT_MODEL,
    *,
    streaming: bool = True,
    enable_portfolio: bool = False,
    portfolio: Optional[Any] = None,
):
    """Build a fresh agent and its (fresh) tool instances. New tools are returned each call
    so their registries start empty -- artifact ids must not leak across unrelated runs.

    Returns (agent, search_tool, portfolio_tool). `portfolio_tool` is None unless
    `enable_portfolio` is set; when set, the heavy quant stack (quant/marketdata via the
    `portfolio` extra) is imported lazily so the research-only path and the Vercel deploy
    never pay for it. If a parsed `portfolio` is supplied, a get_uploaded_portfolio tool is
    added so the agent can analyze "my portfolio".
    """
    chat_model = FixedChatNebius(model=model, streaming=streaming)
    search_tool = TavilySearchWithRegistry()
    tools: List[Any] = [search_tool]
    portfolio_tool = None
    system_prompt = SYSTEM_PROMPT

    if enable_portfolio:
        from portfolio_tool import PortfolioAnalysisTool, build_get_portfolio_tool

        portfolio_tool = PortfolioAnalysisTool()
        tools.append(portfolio_tool)
        if portfolio is not None:
            tools.append(build_get_portfolio_tool(portfolio))
        system_prompt = SYSTEM_PROMPT + PORTFOLIO_PROMPT_SUFFIX

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=system_prompt,
        response_format=ResearchAnswer,
    )
    return agent, search_tool, portfolio_tool


@dataclass
class RunResult:
    question: str
    model: str
    answer: ResearchAnswer
    registry: SourceRegistry
    validation: ValidationResult
    tool_calls: List[dict]
    latency_seconds: float
    run_id: Optional[str] = None
    metric_registry: Optional[Any] = None  # MetricRegistry when the portfolio tool ran
    chart_registry: Optional[Any] = None   # ChartRegistry when the portfolio tool ran

    def to_log_record(self) -> dict:
        return {
            "timestamp": time.time(),
            "question": self.question,
            "model": self.model,
            "answer": self.answer.answer,
            "cited_source_ids": sorted(set(self.answer.cited_source_ids)),
            "num_sources": len(self.registry),
            "num_metrics": len(self.metric_registry) if self.metric_registry is not None else 0,
            "num_charts": len(self.chart_registry) if self.chart_registry is not None else 0,
            "tool_calls": self.tool_calls,
            "latency_seconds": round(self.latency_seconds, 3),
            "validation": self.validation.to_dict(),
            "langsmith_run_id": self.run_id,
        }


def extract_tool_calls(messages: List[Any]) -> List[dict]:
    calls: List[dict] = []
    for message in messages:
        if getattr(message, "type", None) != "ai":
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            calls.append({"name": tool_call.get("name"), "args": tool_call.get("args", {})})
    return calls


def repair_invalid_citations(
    agent,
    search_tool: TavilySearchWithRegistry,
    messages: List[Any],
    structured_response: ResearchAnswer,
    validation: ValidationResult,
    *,
    max_attempts: int = 2,
    metric_registry: Optional[Any] = None,
    chart_registry: Optional[Any] = None,
) -> tuple[ResearchAnswer, ValidationResult, List[Any]]:
    """Give the model a bounded chance to fix its own citation mistakes.

    In testing, gpt-oss-120b sometimes reverts to its pretrained citation habit
    (`【n†source】`-style markers) instead of the requested `[n]` format, even though the
    underlying source ids it meant to cite are valid. Rather than just reporting that
    failure, replay the conversation with one corrective turn telling it exactly what
    was wrong and which ids are valid, and re-validate. This still terminates: if the
    model fails again after `max_attempts`, the (still invalid) result is returned as-is
    -- the badge will show the failure rather than silently accepting it.
    """
    attempts = 0
    while not validation.valid and attempts < max_attempts:
        attempts += 1
        metric_hint = ""
        if metric_registry is not None:
            metric_hint = (
                f" Valid metric ids are {sorted(metric_registry.ids())}; reference them only as "
                "[metric:k]."
            )
        repair_message = {
            "role": "user",
            "content": (
                "Your last answer failed validation: "
                + "; ".join(validation.errors)
                + f". Valid source ids in this conversation are {sorted(search_tool.registry.ids())}."
                + metric_hint
                + " Rewrite the answer using ONLY plain ASCII [n] markers for source citations "
                "(no other citation style), citing only valid ids, and call the ResearchAnswer "
                "tool again."
            ),
        }
        messages = messages + [repair_message]
        state = agent.invoke({"messages": messages})
        candidate = state.get("structured_response") or parse_structured_fallback(
            state.get("messages", [])
        )
        if candidate is None:
            break
        candidate.answer = normalize_citation_markers(candidate.answer)
        structured_response = candidate
        messages = state["messages"]
        validation = validate_artifacts(
            structured_response, search_tool.registry, metric_registry, chart_registry
        )

    return structured_response, validation, messages


def run_query(
    question: str,
    agent=None,
    search_tool=None,
    *,
    model: str = DEFAULT_MODEL,
    log: bool = True,
    repair_attempts: int = 2,
    enable_portfolio: bool = False,
    portfolio_tool=None,
    portfolio: Optional[Any] = None,
) -> RunResult:
    """Run one question through the agent end-to-end (no console streaming) and return
    a structured RunResult. This is the path used by the eval/dataset script and by
    tests; the CLI below wraps the same agent with live streaming for interactive use.

    Set enable_portfolio=True to give the agent the analyze_portfolio tool (quant metrics);
    its MetricRegistry is then validated alongside source citations and attached to the
    RunResult. Set log=False to skip the JSONL/LangSmith side effects (used by unit tests so
    they don't pollute logs/runs.jsonl)."""
    if agent is None or search_tool is None:
        agent, search_tool, portfolio_tool = build_agent(
            model=model, streaming=False, enable_portfolio=enable_portfolio, portfolio=portfolio
        )

    metric_registry = portfolio_tool.registry if portfolio_tool is not None else None
    chart_registry = portfolio_tool.chart_registry if portfolio_tool is not None else None

    start = time.perf_counter()
    with collect_runs() as run_collector:
        state = agent.invoke({"messages": [{"role": "user", "content": question}]})

        agent_messages = state.get("messages", [])
        structured_response = (
            state.get("structured_response")
            or parse_structured_fallback(agent_messages)
            or coerce_research_answer(agent_messages)
        )
        if structured_response is None:
            raise RuntimeError("Agent did not produce a structured ResearchAnswer.")

        structured_response.answer = normalize_citation_markers(structured_response.answer)
        validation = validate_artifacts(
            structured_response, search_tool.registry, metric_registry, chart_registry
        )
        messages = state["messages"]
        if repair_attempts > 0:
            structured_response, validation, messages = repair_invalid_citations(
                agent, search_tool, messages, structured_response, validation,
                max_attempts=repair_attempts, metric_registry=metric_registry,
                chart_registry=chart_registry,
            )
    latency = time.perf_counter() - start

    tool_calls = extract_tool_calls(messages)
    run_id = str(run_collector.traced_runs[0].id) if run_collector.traced_runs else None

    result = RunResult(
        question=question,
        model=model,
        answer=structured_response,
        registry=search_tool.registry,
        validation=validation,
        tool_calls=tool_calls,
        latency_seconds=latency,
        run_id=run_id,
        metric_registry=metric_registry,
        chart_registry=chart_registry,
    )

    if log:
        attach_langsmith_feedback(run_id, validation)
        log_run(result.to_log_record())

    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

app = typer.Typer(add_completion=False)

FINAL_ANSWER_TOOL_NAME = ResearchAnswer.__name__


def require_env(name: str, instructions: str) -> None:
    if os.getenv(name):
        return
    console.print(f"[bold red]Missing {name}[/bold red]")
    console.print(instructions)
    raise typer.Exit(code=1)


def truncate(value: Any, limit: int = 900) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def format_tool_result(content: Any) -> str:
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return truncate(content)

    if not isinstance(payload, dict) or "results" not in payload:
        return truncate(payload)

    lines = [f"Query: {payload.get('query', '')}"]
    if payload.get("duplicate_results_removed"):
        lines.append(f"(deduped {payload['duplicate_results_removed']} duplicate result(s))")
    lines.append("")
    for result in payload.get("results", [])[:5]:
        source_id = result.get("id")
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = " ".join(result.get("content", "").split())
        lines.append(f"[{source_id}] {title}")
        lines.append(f"     {url}")
        if snippet:
            lines.append(f"     {truncate(snippet, limit=220)}")
        lines.append("")
    return "\n".join(lines).strip()


def print_tool_call(name: str, args: Any) -> None:
    console.print()
    console.print(
        Panel(Text(truncate(args, limit=700)), title=f"Tool call: {name}", border_style="yellow")
    )


def print_tool_result(message: Any) -> None:
    name = getattr(message, "name", None) or "tool"
    content = format_tool_result(getattr(message, "content", ""))
    console.print()
    console.print(Panel(Text(content), title=f"Tool result: {name}", border_style="yellow"))


def render_answer(answer: ResearchAnswer) -> None:
    console.print()
    console.print(Panel(Markdown(answer.answer), title="Answer", border_style="green"))


def render_sources(registry: SourceRegistry) -> None:
    table = Table(title="Sources", border_style="blue")
    table.add_column("id", justify="right", style="bold")
    table.add_column("title")
    table.add_column("url", overflow="fold")
    for record in registry.all_sources():
        table.add_row(str(record.id), record.title, record.url)
    console.print(table if len(registry) else Panel("No sources retrieved.", border_style="blue"))


def render_validation_badge(validation: ValidationResult) -> None:
    if validation.valid:
        console.print(Panel("[bold green]✓ All citations verified against retrieved sources[/bold green]", border_style="green"))
    else:
        console.print(
            Panel(
                f"[bold red]✗ Citation validation failed:[/bold red] {'; '.join(validation.errors)}",
                border_style="red",
            )
        )


@app.command()
def main(
    question: Annotated[list[str], typer.Argument(help="Question")],
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
) -> None:
    """Ask a question and stream tavily_maxer: a citation-grounded, traced research agent."""

    require_env(
        "TAVILY_API_KEY",
        "Create one at https://app.tavily.com, then run: export TAVILY_API_KEY='tvly-...'",
    )
    require_env(
        "NEBIUS_API_KEY",
        "Create one at https://tokenfactory.nebius.com, then run: export NEBIUS_API_KEY='...'",
    )

    question_text = " ".join(question)
    agent, search_tool, _portfolio_tool = build_agent(model=model, streaming=True)

    console.print(Panel.fit(question_text, title="Question", border_style="cyan"))
    console.rule("[bold blue]Agent stream")

    tool_buffers: dict[str, dict[str, str]] = {}
    printed_tool_calls: set[str] = set()
    structured_response: Optional[ResearchAnswer] = None
    ai_messages: List[Any] = []
    collected_messages: List[Any] = [{"role": "user", "content": question_text}]
    run_id: Optional[str] = None
    start = time.perf_counter()

    try:
        with collect_runs() as run_collector:
            stream = agent.stream(
                {"messages": [{"role": "user", "content": question_text}]},
                stream_mode=["messages", "updates"],
            )

            for mode, data in stream:
                if mode == "messages":
                    message, _metadata = data
                    if getattr(message, "type", None) == "tool":
                        continue

                    tool_call_chunks = getattr(message, "tool_call_chunks", []) or []
                    for chunk in tool_call_chunks:
                        key = str(chunk.get("id") or chunk.get("index") or "tool_call")
                        buffer = tool_buffers.setdefault(key, {"name": "", "args": ""})
                        if chunk.get("name"):
                            buffer["name"] += chunk["name"]
                        if chunk.get("args"):
                            buffer["args"] += chunk["args"]

                        # The structured final-answer call is rendered specially below
                        # once it's complete; don't stream its raw JSON as a tool call.
                        if buffer["name"] == FINAL_ANSWER_TOOL_NAME:
                            continue

                        if key not in printed_tool_calls and buffer["name"]:
                            printed_tool_calls.add(key)
                            console.print(
                                f"\n[bold yellow]Tool call[/bold yellow] [yellow]{buffer['name']}[/yellow]",
                                highlight=False,
                            )
                            console.print("[dim yellow]args: [/dim yellow]", end="")
                        if chunk.get("args"):
                            console.print(chunk["args"], style="yellow", end="", highlight=False)
                            console.file.flush()

                elif mode == "updates":
                    for node_update in data.values():
                        if node_update.get("structured_response") is not None:
                            structured_response = node_update["structured_response"]

                        for message in node_update.get("messages", []):
                            if getattr(message, "type", None) == "ai":
                                ai_messages.append(message)
                                collected_messages.append(message)
                                for tool_call in getattr(message, "tool_calls", []) or []:
                                    if tool_call.get("name") == FINAL_ANSWER_TOOL_NAME:
                                        continue
                                    key = str(tool_call.get("id") or tool_call.get("name") or "tool_call")
                                    if key not in printed_tool_calls:
                                        printed_tool_calls.add(key)
                                        print_tool_call(tool_call.get("name", "tool"), tool_call.get("args", {}))

                            if getattr(message, "type", None) == "tool":
                                collected_messages.append(message)
                                print_tool_result(message)

            if structured_response is None:
                structured_response = parse_structured_fallback(ai_messages) or coerce_research_answer(ai_messages)

            if structured_response is None:
                console.print("[bold red]Agent did not produce a structured answer.[/bold red]")
                raise typer.Exit(code=1)

            structured_response.answer = normalize_citation_markers(structured_response.answer)
            validation = validate_citations(structured_response, search_tool.registry)
            if not validation.valid:
                console.print("\n[dim yellow]Citation validation failed, asking the model to fix it...[/dim yellow]")
                structured_response, validation, collected_messages = repair_invalid_citations(
                    agent, search_tool, collected_messages, structured_response, validation
                )

            run_id = str(run_collector.traced_runs[0].id) if run_collector.traced_runs else None

        console.print()
        console.rule("[bold blue]Result")

        latency = time.perf_counter() - start

        render_answer(structured_response)
        render_sources(search_tool.registry)
        render_validation_badge(validation)

        attach_langsmith_feedback(run_id, validation)
        log_run(
            RunResult(
                question=question_text,
                model=model,
                answer=structured_response,
                registry=search_tool.registry,
                validation=validation,
                tool_calls=extract_tool_calls(collected_messages),
                latency_seconds=latency,
                run_id=run_id,
            ).to_log_record()
        )

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"\n[bold red]Agent run failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
