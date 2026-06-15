"""Tests for the verification gate (deterministic layer; no LLM)."""

from grounded.schemas import Citation, Claim, DraftAnswer, Source, Verdict
from grounded.verify import assemble_ledger, quote_supported, verify_claims

SRC = Source(
    id="S1",
    url="http://example.com",
    title="Eiffel Tower",
    content="The Eiffel Tower is 330 metres tall and is located in Paris, France.",
)


def test_quote_supported_exact_and_normalized():
    assert quote_supported("330 metres tall", SRC.content)
    assert quote_supported("THE eiffel   tower is 330 metres tall", SRC.content)


def test_quote_supported_rejects_fabrication():
    assert not quote_supported("the tower is 500 metres tall", SRC.content)
    assert not quote_supported("", SRC.content)


def _draft() -> DraftAnswer:
    return DraftAnswer(
        summary="The Eiffel Tower is 330m tall [S1].",
        claims=[
            Claim(text="It is 330 metres tall", citations=[Citation(source_id="S1", quote="330 metres tall")]),
            Claim(text="It is 500 metres tall", citations=[Citation(source_id="S1", quote="500 metres tall")]),
            Claim(text="It was the tallest until 1930", citations=[]),
        ],
    )


def test_verify_flags_fabricated_quote_and_missing_citation():
    vs = verify_claims(_draft(), [SRC], judge=None)
    assert vs[0].quote_found and vs[0].verdict == Verdict.supported
    assert not vs[1].quote_found and vs[1].verdict == Verdict.unsupported
    assert not vs[2].quote_found and vs[2].verdict == Verdict.unsupported


def test_assemble_ledger_scores_and_caveats():
    draft = _draft()
    vs = verify_claims(draft, [SRC], judge=None)
    ledger = assemble_ledger("How tall is it?", draft, [SRC], vs, min_groundedness=0.6)
    assert abs(ledger.groundedness - 1 / 3) < 0.01  # 1 of 3 supported
    assert "It is 500 metres tall" in ledger.flagged_claims
    assert "⚠ Unverified" in ledger.answer
    assert ledger.sources == [SRC]
