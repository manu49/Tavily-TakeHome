"""Data model for the research pipeline and the evidence ledger.

These Pydantic models are the contract between every stage:

    question -> SearchPlan -> [Source] -> DraftAnswer -> EvidenceLedger

The `EvidenceLedger` is the deliverable that makes an answer auditable: for
every claim it records the source it leans on, the verbatim supporting quote,
whether that quote was actually found in the source, and a verification verdict.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
class Topic(str, Enum):
    """Tavily search topic. Routing the query to the right index materially
    improves recall: `news`/`finance` are time-sensitive and freshness-ranked."""

    general = "general"
    news = "news"
    finance = "finance"


class SubQuery(BaseModel):
    """One focused search, routed to the most appropriate Tavily index."""

    query: str = Field(description="A focused, self-contained search query.")
    topic: Topic = Field(
        default=Topic.general,
        description="Tavily index to route to: general, news, or finance.",
    )
    time_range: str | None = Field(
        default=None,
        description="Optional recency filter: one of day, week, month, year.",
    )
    rationale: str | None = Field(
        default=None, description="Why this sub-query helps answer the question."
    )


class SearchPlan(BaseModel):
    """The decomposition of the user's question into searchable sub-queries."""

    subqueries: list[SubQuery] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Retrieval corpus
# --------------------------------------------------------------------------- #
class Source(BaseModel):
    """A single retrieved source with its full extracted text.

    `id` is a short, stable handle (S1, S2, ...) used for inline citations so
    the model never has to reproduce long URLs."""

    id: str
    url: str
    title: str = ""
    content: str = ""  # full extracted text — the grounding corpus
    score: float | None = None  # Tavily relevance score, when available


# --------------------------------------------------------------------------- #
# Synthesis (structured model output)
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A pointer from a claim to the exact passage that backs it."""

    source_id: str = Field(description="The Source.id this claim relies on, e.g. 'S2'.")
    quote: str = Field(
        description="A short VERBATIM span copied from that source's text "
        "that directly supports the claim."
    )


class Claim(BaseModel):
    """One atomic, checkable assertion in the answer."""

    text: str = Field(description="A single factual statement.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources + verbatim quotes that support this claim.",
    )


class DraftAnswer(BaseModel):
    """The model's structured answer: prose plus the claims it is built on."""

    summary: str = Field(
        description="The answer in prose, with inline [S1]/[S2] source markers."
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="The atomic claims the summary asserts, each with citations.",
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    supported = "supported"  # the source entails the claim
    partial = "partial"  # related but the claim overreaches
    unsupported = "unsupported"  # the source does not support the claim


class ClaimVerification(BaseModel):
    """The audit result for one claim."""

    claim: Claim
    quote_found: bool = Field(
        description="Deterministic check: does the cited quote actually appear "
        "in the cited source's text? Catches fabricated quotes for free."
    )
    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0, description="Groundedness in [0, 1].")
    reason: str = ""


class EvidenceLedger(BaseModel):
    """The auditable record returned to the caller alongside the answer."""

    question: str
    answer: str  # final answer, with unsupported claims removed or flagged
    sources: list[Source] = Field(default_factory=list)
    verifications: list[ClaimVerification] = Field(default_factory=list)
    groundedness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of claims that passed verification, weighted by score.",
    )
    flagged_claims: list[str] = Field(
        default_factory=list,
        description="Claims that failed verification and were withheld or caveated.",
    )
