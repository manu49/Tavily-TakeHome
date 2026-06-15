"""A reference-free, mode-agnostic answer judge for the evaluation.

This judge is deliberately *independent* of the pipeline's own verification
gate: it re-derives the claims in a final answer and checks them against the
sources that answer had available. Using one external judge for both the
grounded pipeline and the baseline makes the comparison apples-to-apples, and
keeps measurement separate from the component being measured.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from grounded.config import Settings, build_chat_model
from grounded.schemas import Source


class AnswerEval(BaseModel):
    total_claims: int = Field(ge=0, description="Distinct factual claims in the answer.")
    supported_claims: int = Field(ge=0, description="Claims fully supported by the sources.")
    unsupported: list[str] = Field(default_factory=list)
    citation_quality: float = Field(
        ge=0.0, le=1.0, description="Are claims clearly attributable to specific sources?"
    )
    notes: str = ""

    @property
    def groundedness(self) -> float:
        return self.supported_claims / self.total_claims if self.total_claims else 0.0


_JUDGE_SYSTEM = (
    "You are an impartial evaluator of factual grounding. You are given a "
    "QUESTION, an ANSWER, and the SOURCES that answer had access to.\n"
    "1. Extract the distinct factual claims the ANSWER makes.\n"
    "2. Count how many are FULLY supported by the SOURCES.\n"
    "3. List any claims that are not supported.\n"
    "4. Rate citation_quality in [0,1]: are claims clearly tied to specific "
    "sources, or asserted without attribution?\n"
    "Judge ONLY against the provided sources. Do not use outside knowledge. "
    "Ignore meta-text such as 'unverified' caveats when extracting claims."
)


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n]


def judge_answer(
    settings: Settings,
    question: str,
    answer: str,
    sources: list[Source],
    char_budget: int = 4000,
) -> AnswerEval:
    block = "\n\n".join(
        f"[{s.id}] {s.title} ({s.url})\n{_clip(s.content, char_budget)}" for s in sources
    )
    model = build_chat_model(settings, for_judge=True, max_tokens=1500).with_structured_output(
        AnswerEval
    )
    return model.invoke(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{block}",
            },
        ]
    )
