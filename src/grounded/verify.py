"""The verification gate — the heart of the "verifiable" claim.

Two layers, cheap-first:

  1. Deterministic quote check (no LLM): does the verbatim quote the model
     attached to a claim actually appear in the cited source's text? This
     catches fabricated quotes for free and is fully unit-testable.

  2. LLM entailment check (only when the quote is real): does the quote
     actually *support* the claim, or did the model cite a real passage that
     says something else? This catches the subtler failure mode.

Claims that fail either layer are flagged, and the final answer caveats them
instead of asserting them. The whole thing is summarized as a groundedness
score in the EvidenceLedger.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from opentelemetry import trace
from pydantic import BaseModel, Field

from .schemas import (
    Claim,
    ClaimVerification,
    DraftAnswer,
    EvidenceLedger,
    Source,
    Verdict,
)

tracer = trace.get_tracer("grounded.verify")

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text or "").strip().lower()


def quote_supported(quote: str, source_text: str, threshold: float = 0.92) -> bool:
    """True if `quote` appears (near-)verbatim in `source_text`.

    Exact normalized-substring match first; a fuzzy fallback then tolerates
    minor punctuation/whitespace drift from extraction without accepting a
    quote the source never made.
    """
    q, src = _normalize(quote), _normalize(source_text)
    if not q:
        return False
    if q in src:
        return True
    # Fuzzy fallback: slide a window the size of the quote across the source.
    if len(q) > len(src):
        return False
    best = 0.0
    step = max(1, len(q) // 4)
    for start in range(0, len(src) - len(q) + 1, step):
        window = src[start : start + len(q)]
        ratio = SequenceMatcher(None, q, window).ratio()
        if ratio >= threshold:
            return True
        best = max(best, ratio)
    return best >= threshold


class EntailmentJudgment(BaseModel):
    """Structured verdict from the LLM judge for one claim."""

    verdict: Verdict = Field(description="supported | partial | unsupported")
    score: float = Field(ge=0.0, le=1.0, description="Groundedness in [0,1].")
    reason: str = Field(description="One sentence justifying the verdict.")


_JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. Given a CLAIM and the exact SOURCE "
    "PASSAGES it cites, decide whether the passages support the claim.\n"
    "- 'supported': the passages directly entail the claim.\n"
    "- 'partial': related but the claim overreaches or adds unsupported detail.\n"
    "- 'unsupported': the passages do not establish the claim.\n"
    "Judge only against the passages provided. Do not use outside knowledge."
)


def _judge_claim(claim: Claim, source_by_id: dict[str, Source], judge) -> EntailmentJudgment:
    passages = []
    for cite in claim.citations:
        src = source_by_id.get(cite.source_id)
        title = src.title if src else cite.source_id
        passages.append(f"[{cite.source_id}] {title}\n\"{cite.quote}\"")
    human = (
        f"CLAIM:\n{claim.text}\n\nSOURCE PASSAGES:\n" + "\n\n".join(passages)
    )
    return judge.invoke(
        [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": human}]
    )


def verify_claims(
    draft: DraftAnswer,
    sources: list[Source],
    judge=None,
) -> list[ClaimVerification]:
    """Verify every claim. `judge` is a structured-output runnable returning
    EntailmentJudgment; pass None for deterministic-only mode (quote check)."""
    source_by_id = {s.id: s for s in sources}
    verifications: list[ClaimVerification] = []

    with tracer.start_as_current_span("verify.claims") as span:
        span.set_attribute("verify.claim_count", len(draft.claims))
        for claim in draft.claims:
            quote_found = bool(claim.citations) and all(
                (s := source_by_id.get(c.source_id)) is not None
                and quote_supported(c.quote, s.content)
                for c in claim.citations
            )

            if not quote_found:
                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        quote_found=False,
                        verdict=Verdict.unsupported,
                        score=0.0,
                        reason="No citation, or the cited quote was not found in "
                        "the source (possible fabrication).",
                    )
                )
                continue

            if judge is None:
                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        quote_found=True,
                        verdict=Verdict.supported,
                        score=1.0,
                        reason="Quote verified in source (deterministic mode).",
                    )
                )
                continue

            j = _judge_claim(claim, source_by_id, judge)
            verifications.append(
                ClaimVerification(
                    claim=claim,
                    quote_found=True,
                    verdict=j.verdict,
                    score=j.score,
                    reason=j.reason,
                )
            )

        grounded = sum(1 for v in verifications if v.verdict == Verdict.supported)
        span.set_attribute("verify.supported_count", grounded)
    return verifications


def assemble_ledger(
    question: str,
    draft: DraftAnswer,
    sources: list[Source],
    verifications: list[ClaimVerification],
    min_groundedness: float = 0.6,
) -> EvidenceLedger:
    """Compute the groundedness score, flag weak claims, and caveat the answer."""
    scores = [v.score for v in verifications]
    groundedness = sum(scores) / len(scores) if scores else 0.0

    flagged = [
        v.claim.text
        for v in verifications
        if v.verdict != Verdict.supported or v.score < min_groundedness
    ]

    answer = draft.summary
    if flagged:
        bullets = "\n".join(f"  - {c}" for c in flagged)
        answer += (
            "\n\n⚠ Unverified — the following claims are not fully supported by "
            f"the retrieved sources and should be treated with caution:\n{bullets}"
        )

    return EvidenceLedger(
        question=question,
        answer=answer,
        sources=sources,
        verifications=verifications,
        groundedness=round(groundedness, 3),
        flagged_claims=flagged,
    )
