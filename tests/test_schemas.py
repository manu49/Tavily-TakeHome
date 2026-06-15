"""Schema sanity + JSON round-trip for the evidence ledger."""

import json

from grounded.schemas import (
    Claim,
    Citation,
    DraftAnswer,
    EvidenceLedger,
    SearchPlan,
    SubQuery,
    Topic,
)


def test_subquery_defaults_to_general_topic():
    sq = SubQuery(query="what is tavily")
    assert sq.topic == Topic.general
    assert sq.time_range is None


def test_searchplan_requires_a_subquery():
    plan = SearchPlan(subqueries=[SubQuery(query="x")])
    assert len(plan.subqueries) == 1


def test_evidence_ledger_json_roundtrip():
    ledger = EvidenceLedger(
        question="q",
        answer="a",
        verifications=[],
    )
    payload = json.loads(ledger.model_dump_json())
    assert payload["question"] == "q"
    assert payload["groundedness"] == 0.0
    assert payload["flagged_claims"] == []


def test_draft_answer_carries_claims_and_citations():
    draft = DraftAnswer(
        summary="s [S1]",
        claims=[Claim(text="c", citations=[Citation(source_id="S1", quote="q")])],
    )
    assert draft.claims[0].citations[0].source_id == "S1"
