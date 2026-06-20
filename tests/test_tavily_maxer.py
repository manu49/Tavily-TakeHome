"""Extensive test suite for tavily_maxer.py.

Everything here runs offline: no real Tavily or Nebius API calls. Network-shaped
boundaries (TavilySearch.api_wrapper.raw_results, the chat model) are replaced with
deterministic fakes so the suite is fast and repeatable. The live agent itself is
separately exercised by evals/run_samples.py against real APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typer.testing import CliRunner

import tavily_maxer as tm
from artifacts import ChartRegistry, MetricRegistry

runner = CliRunner()


# --------------------------------------------------------------------------------------
# normalize_url
# --------------------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_identical_urls_match(self):
        assert tm.normalize_url("https://example.com/page") == tm.normalize_url(
            "https://example.com/page"
        )

    def test_http_and_https_collapse(self):
        assert tm.normalize_url("http://example.com/page") == tm.normalize_url(
            "https://example.com/page"
        )

    def test_www_prefix_collapses(self):
        assert tm.normalize_url("https://www.example.com/page") == tm.normalize_url(
            "https://example.com/page"
        )

    def test_trailing_slash_collapses(self):
        assert tm.normalize_url("https://example.com/page/") == tm.normalize_url(
            "https://example.com/page"
        )

    def test_root_path_normalizes_to_slash(self):
        assert tm.normalize_url("https://example.com") == tm.normalize_url(
            "https://example.com/"
        )

    def test_utm_params_are_stripped(self):
        tracked = "https://example.com/page?utm_source=newsletter&utm_campaign=x"
        bare = "https://example.com/page"
        assert tm.normalize_url(tracked) == tm.normalize_url(bare)

    def test_fragment_is_stripped(self):
        assert tm.normalize_url("https://example.com/page#section-2") == tm.normalize_url(
            "https://example.com/page"
        )

    def test_meaningful_query_params_preserved(self):
        a = tm.normalize_url("https://example.com/search?q=foo")
        b = tm.normalize_url("https://example.com/search?q=bar")
        assert a != b

    def test_query_param_order_does_not_matter(self):
        assert tm.normalize_url("https://example.com/p?a=1&b=2") == tm.normalize_url(
            "https://example.com/p?b=2&a=1"
        )

    def test_different_paths_do_not_collapse(self):
        assert tm.normalize_url("https://example.com/a") != tm.normalize_url(
            "https://example.com/b"
        )

    def test_host_is_case_insensitive(self):
        assert tm.normalize_url("https://Example.com/Page") == tm.normalize_url(
            "https://example.com/Page"
        )


# --------------------------------------------------------------------------------------
# SourceRegistry
# --------------------------------------------------------------------------------------

class TestSourceRegistry:
    def test_assigns_sequential_ids_starting_at_one(self):
        registry = tm.SourceRegistry()
        labeled, _ = registry.register(
            [{"title": "A", "url": "https://a.com"}, {"title": "B", "url": "https://b.com"}]
        )
        assert [r["id"] for r in labeled] == [1, 2]

    def test_dedupes_duplicate_url_within_one_call(self):
        registry = tm.SourceRegistry()
        labeled, duplicates = registry.register(
            [
                {"title": "A", "url": "https://a.com"},
                {"title": "A again", "url": "https://a.com/"},
            ]
        )
        assert len(labeled) == 1
        assert duplicates == 1
        assert len(registry) == 1

    def test_dedupes_same_url_across_multiple_calls(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "A", "url": "https://a.com"}])
        labeled, duplicates = registry.register([{"title": "A", "url": "https://a.com"}])
        assert duplicates == 1
        assert labeled[0]["id"] == 1
        assert len(registry) == 1

    def test_first_seen_title_wins_on_repeat(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "Original Title", "url": "https://a.com"}])
        labeled, _ = registry.register([{"title": "Different Title", "url": "https://a.com"}])
        assert labeled[0]["title"] == "Original Title"

    def test_results_missing_url_are_skipped(self):
        registry = tm.SourceRegistry()
        labeled, duplicates = registry.register([{"title": "No URL"}, {"title": "", "url": ""}])
        assert labeled == []
        assert duplicates == 0
        assert len(registry) == 0

    def test_ids_persist_across_unrelated_new_sources(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "A", "url": "https://a.com"}])
        labeled, _ = registry.register([{"title": "B", "url": "https://b.com"}])
        assert labeled[0]["id"] == 2

    def test_get_returns_record_by_id(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "A", "url": "https://a.com"}])
        record = registry.get(1)
        assert record is not None
        assert record.url == "https://a.com"

    def test_get_returns_none_for_unknown_id(self):
        registry = tm.SourceRegistry()
        assert registry.get(999) is None

    def test_ids_returns_full_set(self):
        registry = tm.SourceRegistry()
        registry.register(
            [{"title": "A", "url": "https://a.com"}, {"title": "B", "url": "https://b.com"}]
        )
        assert registry.ids() == {1, 2}

    def test_all_sources_sorted_by_id(self):
        registry = tm.SourceRegistry()
        registry.register(
            [{"title": "A", "url": "https://a.com"}, {"title": "B", "url": "https://b.com"}]
        )
        assert [r.id for r in registry.all_sources()] == [1, 2]

    def test_to_markdown_empty_registry(self):
        registry = tm.SourceRegistry()
        assert "No sources" in registry.to_markdown()

    def test_to_markdown_lists_every_source(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "A", "url": "https://a.com"}])
        markdown = registry.to_markdown()
        assert "[1]" in markdown
        assert "https://a.com" in markdown

    def test_len_reflects_unique_source_count(self):
        registry = tm.SourceRegistry()
        registry.register(
            [
                {"title": "A", "url": "https://a.com"},
                {"title": "A dup", "url": "https://a.com"},
                {"title": "B", "url": "https://b.com"},
            ]
        )
        assert len(registry) == 2


# --------------------------------------------------------------------------------------
# TavilySearchWithRegistry
# --------------------------------------------------------------------------------------

class TestTavilySearchWithRegistry:
    @staticmethod
    def make_tool(monkeypatch, raw_results: dict):
        """TavilySearchAPIWrapper is a pydantic model with no `raw_results` field, so it
        rejects instance-attribute assignment. Patch the method on its class instead
        (monkeypatch auto-restores it after the test)."""
        tool = tm.TavilySearchWithRegistry()
        monkeypatch.setattr(
            type(tool.api_wrapper), "raw_results", lambda self, **kwargs: raw_results
        )
        return tool

    def test_labels_results_with_stable_ids(self, monkeypatch):
        tool = self.make_tool(
            monkeypatch,
            {
                "query": "q",
                "results": [
                    {"title": "A", "url": "https://a.com", "content": "..."},
                    {"title": "B", "url": "https://b.com", "content": "..."},
                ],
            },
        )
        output = tool._run(query="q")
        assert [r["id"] for r in output["results"]] == [1, 2]

    def test_dedupes_duplicate_results_in_one_response(self, monkeypatch):
        tool = self.make_tool(
            monkeypatch,
            {
                "query": "q",
                "results": [
                    {"title": "A", "url": "https://a.com"},
                    {"title": "A", "url": "https://a.com/"},
                ],
            },
        )
        output = tool._run(query="q")
        assert len(output["results"]) == 1
        assert output["duplicate_results_removed"] == 1

    def test_registry_persists_across_repeated_tool_calls(self, monkeypatch):
        tool = self.make_tool(
            monkeypatch, {"query": "q", "results": [{"title": "A", "url": "https://a.com"}]}
        )
        tool._run(query="first query")
        monkeypatch.setattr(
            type(tool.api_wrapper),
            "raw_results",
            lambda self, **kwargs: {
                "query": "q2",
                "results": [{"title": "B", "url": "https://b.com"}],
            },
        )
        output = tool._run(query="second query")
        assert output["results"][0]["id"] == 2
        assert len(tool.registry) == 2

    def test_includes_citation_instruction_note(self, monkeypatch):
        tool = self.make_tool(
            monkeypatch, {"query": "q", "results": [{"title": "A", "url": "https://a.com"}]}
        )
        output = tool._run(query="q")
        assert "id" in output["note"]

    def test_other_top_level_payload_keys_are_preserved(self, monkeypatch):
        tool = self.make_tool(
            monkeypatch,
            {
                "query": "q",
                "answer": "a short generated answer",
                "response_time": 1.23,
                "results": [{"title": "A", "url": "https://a.com"}],
            },
        )
        output = tool._run(query="q")
        assert output["answer"] == "a short generated answer"
        assert output["response_time"] == 1.23
        assert output["duplicate_results_removed"] == 0

    def test_fresh_tool_instance_has_empty_registry(self):
        tool = tm.TavilySearchWithRegistry()
        assert len(tool.registry) == 0


# --------------------------------------------------------------------------------------
# Citation validation
# --------------------------------------------------------------------------------------

class TestValidateCitations:
    def make_registry(self, n: int) -> tm.SourceRegistry:
        registry = tm.SourceRegistry()
        registry.register([{"title": f"S{i}", "url": f"https://s{i}.com"} for i in range(1, n + 1)])
        return registry

    def test_valid_when_all_citations_resolve(self):
        registry = self.make_registry(2)
        answer = tm.ResearchAnswer(answer="Paris is the capital [1][2].", cited_source_ids=[1, 2])
        result = tm.validate_citations(answer, registry)
        assert result.valid
        assert result.errors == []

    def test_valid_with_no_citations_at_all(self):
        registry = self.make_registry(0)
        answer = tm.ResearchAnswer(answer="No sources needed for this.", cited_source_ids=[])
        result = tm.validate_citations(answer, registry)
        assert result.valid

    def test_invalid_when_declared_id_not_in_registry(self):
        registry = self.make_registry(1)
        answer = tm.ResearchAnswer(answer="Claim [1].", cited_source_ids=[1, 99])
        result = tm.validate_citations(answer, registry)
        assert not result.valid
        assert 99 in result.missing_ids

    def test_invalid_when_inline_marker_not_in_registry(self):
        registry = self.make_registry(1)
        answer = tm.ResearchAnswer(answer="Claim [1] and also [7].", cited_source_ids=[1])
        result = tm.validate_citations(answer, registry)
        assert not result.valid
        assert 7 in result.missing_ids

    def test_detects_stray_harmony_style_markers(self):
        registry = self.make_registry(1)
        answer = tm.ResearchAnswer(
            answer="Paris is the capital【1†source】.", cited_source_ids=[1]
        )
        result = tm.validate_citations(answer, registry)
        assert not result.valid
        assert any("Non-standard" in e for e in result.errors)

    def test_stray_marker_fails_even_if_declared_ids_all_valid(self):
        registry = self.make_registry(1)
        answer = tm.ResearchAnswer(
            answer="Claim [1] plus a stray 【0†L1-L2】 marker.",
            cited_source_ids=[1],
        )
        result = tm.validate_citations(answer, registry)
        assert not result.valid

    def test_inline_ids_extracted_correctly(self):
        registry = self.make_registry(3)
        answer = tm.ResearchAnswer(answer="See [1], [2], and [3].", cited_source_ids=[1, 2, 3])
        result = tm.validate_citations(answer, registry)
        assert result.inline_ids == {1, 2, 3}

    def test_declared_ids_not_used_inline_are_not_an_error(self):
        registry = self.make_registry(2)
        answer = tm.ResearchAnswer(answer="Only inline cite [1].", cited_source_ids=[1, 2])
        result = tm.validate_citations(answer, registry)
        assert result.valid

    def test_to_dict_is_json_serializable(self):
        registry = self.make_registry(1)
        answer = tm.ResearchAnswer(answer="Claim [1].", cited_source_ids=[1])
        result = tm.validate_citations(answer, registry)
        json.dumps(result.to_dict())  # must not raise


# --------------------------------------------------------------------------------------
# normalize_citation_markers (gpt-oss full-width 【n】 -> [n])
# --------------------------------------------------------------------------------------

class TestNormalizeCitationMarkers:
    def test_single_fullwidth_marker(self):
        assert tm.normalize_citation_markers("Paris 【1】.") == "Paris [1]."

    def test_multiple_markers(self):
        assert tm.normalize_citation_markers("a 【1】 b 【3】 c 【5】") == "a [1] b [3] c [5]"

    def test_comma_list_expands(self):
        assert tm.normalize_citation_markers("see 【1, 3】") == "see [1][3]"

    def test_ascii_markers_untouched(self):
        assert tm.normalize_citation_markers("already [1] fine") == "already [1] fine"

    def test_structured_marker_left_for_validation_to_flag(self):
        # 【1†L8-L15】 has extra structure -> NOT normalized, so validation still catches it.
        text = "claim 【1†L8-L15】"
        assert tm.normalize_citation_markers(text) == text

    def test_normalized_text_then_passes_validation(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "S1", "url": "https://e.com/1"}])
        answer = tm.ResearchAnswer(answer=tm.normalize_citation_markers("Fact 【1】."), cited_source_ids=[1])
        assert tm.validate_citations(answer, registry).valid


# --------------------------------------------------------------------------------------
# validate_artifacts (sources + metrics)
# --------------------------------------------------------------------------------------

class TestValidateArtifacts:
    def make_metric_registry(self, n: int):
        reg = MetricRegistry()
        for i in range(n):
            reg.register(f"k{i}", f"Metric {i}", 0.1 * i, "ratio", "def")
        return reg

    def make_source_registry(self, n: int):
        registry = tm.SourceRegistry()
        registry.register([{"title": f"S{i}", "url": f"https://e.com/{i}"} for i in range(n)])
        return registry

    def test_valid_sources_and_metrics_together(self):
        answer = tm.ResearchAnswer(
            answer="Fact [1] and Sharpe is high [metric:1].",
            cited_source_ids=[1],
            referenced_metric_ids=[1],
        )
        result = tm.validate_artifacts(answer, self.make_source_registry(1), self.make_metric_registry(1))
        assert result.valid
        assert result.inline_metric_ids == {1}

    def test_inline_metric_markers_are_detected(self):
        answer = tm.ResearchAnswer(answer="See [metric:2] and [metric:3].")
        result = tm.validate_artifacts(answer, tm.SourceRegistry(), self.make_metric_registry(5))
        assert result.inline_metric_ids == {2, 3}
        assert result.valid

    def test_missing_metric_id_fails_but_isolates_from_sources(self):
        answer = tm.ResearchAnswer(answer="Good source [1], bad metric [metric:9].", cited_source_ids=[1])
        result = tm.validate_artifacts(answer, self.make_source_registry(1), self.make_metric_registry(2))
        assert not result.valid
        assert result.missing_ids == set()           # source side is fine
        assert result.missing_metric_ids == {9}       # metric side caught it

    def test_metric_marker_does_not_trip_source_citation_regex(self):
        # `[metric:3]` must NOT be read as source citation `[3]`.
        answer = tm.ResearchAnswer(answer="Only a metric here [metric:3].")
        result = tm.validate_artifacts(answer, tm.SourceRegistry(), self.make_metric_registry(3))
        assert result.inline_ids == set()
        assert result.inline_metric_ids == {3}

    def make_chart_registry(self, n: int):
        reg = ChartRegistry()
        for i in range(n):
            reg.register("kind", f"Chart {i}", "cap", {"data": [], "layout": {}})
        return reg

    def test_valid_chart_reference_passes(self):
        answer = tm.ResearchAnswer(answer="See the chart [chart:1].", referenced_chart_ids=[1])
        result = tm.validate_artifacts(
            answer, tm.SourceRegistry(), None, self.make_chart_registry(2)
        )
        assert result.valid
        assert result.inline_chart_ids == {1}

    def test_missing_chart_id_fails(self):
        answer = tm.ResearchAnswer(answer="Bad embed [chart:9].")
        result = tm.validate_artifacts(
            answer, tm.SourceRegistry(), None, self.make_chart_registry(2)
        )
        assert not result.valid
        assert result.missing_chart_ids == {9}

    def test_chart_reference_without_registry_is_invalid(self):
        answer = tm.ResearchAnswer(answer="Sneaky [chart:1].")
        result = tm.validate_citations(answer, tm.SourceRegistry())
        assert not result.valid
        assert 1 in result.missing_chart_ids

    def test_chart_marker_isolated_from_metric_and_source(self):
        answer = tm.ResearchAnswer(answer="[1] [metric:1] [chart:1] together.")
        result = tm.validate_artifacts(
            answer, self.make_source_registry(1), self.make_metric_registry(1),
            self.make_chart_registry(1),
        )
        assert result.inline_ids == {1}
        assert result.inline_metric_ids == {1}
        assert result.inline_chart_ids == {1}
        assert result.valid


# --------------------------------------------------------------------------------------
# Observability: JSONL logging + LangSmith feedback gating
# --------------------------------------------------------------------------------------

class TestLogRun:
    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "runs.jsonl"
        tm.log_run({"a": 1}, path=path)
        assert path.exists()

    def test_appends_one_json_line_per_call(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        tm.log_run({"a": 1}, path=path)
        tm.log_run({"a": 2}, path=path)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["a"] == 1
        assert json.loads(lines[1])["a"] == 2

    def test_non_json_native_values_are_stringified_not_raised(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        tm.log_run({"ids": {1, 2, 3}}, path=path)  # sets aren't natively JSON-serializable
        record = json.loads(path.read_text())
        assert isinstance(record["ids"], str)


class TestLangsmithTracingEnabled:
    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("LANGSMITH_TRACING", value)
        assert tm.langsmith_tracing_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", ""])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("LANGSMITH_TRACING", value)
        assert tm.langsmith_tracing_enabled() is False

    def test_unset_is_falsy(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        assert tm.langsmith_tracing_enabled() is False


class TestAttachLangsmithFeedback:
    def test_noop_when_tracing_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        class ShouldNotBeConstructed:
            def __init__(self):
                raise AssertionError("Client() must not be built when tracing is disabled")

        monkeypatch.setattr("langsmith.Client", ShouldNotBeConstructed, raising=False)
        tm.attach_langsmith_feedback("some-run-id", tm.ValidationResult(valid=True))

    def test_noop_when_run_id_missing(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        tm.attach_langsmith_feedback(None, tm.ValidationResult(valid=True))  # must not raise

    def test_swallows_client_errors(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        class BoomClient:
            def __init__(self):
                raise RuntimeError("network down")

        monkeypatch.setattr("langsmith.Client", BoomClient, raising=False)
        # Should not raise even though Client() construction blows up.
        tm.attach_langsmith_feedback("run-id", tm.ValidationResult(valid=False, errors=["x"]))


# --------------------------------------------------------------------------------------
# parse_structured_fallback (Nebius tool_choice="required" recovery path)
# --------------------------------------------------------------------------------------

class TestParseStructuredFallback:
    def test_recovers_valid_json_text_answer(self):
        messages = [
            HumanMessage(content="q"),
            AIMessage(content='{"answer": "Paris [1].", "cited_source_ids": [1]}'),
        ]
        result = tm.parse_structured_fallback(messages)
        assert isinstance(result, tm.ResearchAnswer)
        assert result.cited_source_ids == [1]

    def test_returns_none_for_non_json_text(self):
        messages = [AIMessage(content="Sorry, I don't know.")]
        assert tm.parse_structured_fallback(messages) is None

    def test_returns_none_for_malformed_json(self):
        messages = [AIMessage(content="{not valid json")]
        assert tm.parse_structured_fallback(messages) is None

    def test_returns_none_when_no_ai_messages(self):
        messages = [HumanMessage(content="q"), ToolMessage(content="result", tool_call_id="1")]
        assert tm.parse_structured_fallback(messages) is None

    def test_returns_none_for_empty_message_list(self):
        assert tm.parse_structured_fallback([]) is None

    def test_only_inspects_the_last_ai_message(self):
        messages = [
            AIMessage(content='{"answer": "old", "cited_source_ids": []}'),
            ToolMessage(content="result", tool_call_id="1"),
            AIMessage(content="not json, a tool call happened instead"),
        ]
        assert tm.parse_structured_fallback(messages) is None


# --------------------------------------------------------------------------------------
# coerce_research_answer (graceful degradation when the model answers in plain prose)
# --------------------------------------------------------------------------------------

class TestCoerceResearchAnswer:
    def test_wraps_plain_prose_and_lifts_inline_markers(self):
        messages = [AIMessage(content="SpaceX is private, so it has no public stock [2][5].")]
        result = tm.coerce_research_answer(messages)
        assert isinstance(result, tm.ResearchAnswer)
        assert result.answer.startswith("SpaceX is private")
        assert result.cited_source_ids == [2, 5]

    def test_prose_without_markers_yields_empty_citations(self):
        result = tm.coerce_research_answer([AIMessage(content="I could not find a source.")])
        assert isinstance(result, tm.ResearchAnswer)
        assert result.cited_source_ids == []

    def test_recovers_fenced_json(self):
        messages = [AIMessage(content='```json\n{"answer": "Paris [1].", "cited_source_ids": [1]}\n```')]
        result = tm.coerce_research_answer(messages)
        assert isinstance(result, tm.ResearchAnswer)
        assert result.answer == "Paris [1]."
        assert result.cited_source_ids == [1]

    def test_skips_tool_call_only_turns_for_the_final_prose(self):
        messages = [
            AIMessage(content="", tool_calls=[{"name": "tavily_search", "args": {}, "id": "1"}]),
            ToolMessage(content="result", tool_call_id="1"),
            AIMessage(content="Final prose answer [3]."),
        ]
        result = tm.coerce_research_answer(messages)
        assert result.answer == "Final prose answer [3]."
        assert result.cited_source_ids == [3]

    def test_returns_none_when_no_ai_text(self):
        assert tm.coerce_research_answer([HumanMessage(content="q")]) is None
        assert tm.coerce_research_answer([]) is None


# --------------------------------------------------------------------------------------
# extract_tool_calls
# --------------------------------------------------------------------------------------

class TestExtractToolCalls:
    def test_extracts_name_and_args_from_ai_messages(self):
        messages = [
            HumanMessage(content="q"),
            AIMessage(
                content="",
                tool_calls=[{"name": "tavily_search", "args": {"query": "x"}, "id": "1"}],
            ),
        ]
        calls = tm.extract_tool_calls(messages)
        assert calls == [{"name": "tavily_search", "args": {"query": "x"}}]

    def test_ignores_non_ai_messages(self):
        messages = [HumanMessage(content="q"), ToolMessage(content="r", tool_call_id="1")]
        assert tm.extract_tool_calls(messages) == []

    def test_ai_message_without_tool_calls_contributes_nothing(self):
        messages = [AIMessage(content="just text")]
        assert tm.extract_tool_calls(messages) == []

    def test_multiple_tool_calls_across_messages_are_all_collected(self):
        messages = [
            AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}]),
            ToolMessage(content="r1", tool_call_id="1"),
            AIMessage(content="", tool_calls=[{"name": "b", "args": {}, "id": "2"}]),
        ]
        calls = tm.extract_tool_calls(messages)
        assert [c["name"] for c in calls] == ["a", "b"]


# --------------------------------------------------------------------------------------
# RunResult.to_log_record
# --------------------------------------------------------------------------------------

class TestRunResultLogRecord:
    def test_record_shape_and_rounding(self):
        registry = tm.SourceRegistry()
        registry.register([{"title": "A", "url": "https://a.com"}])
        answer = tm.ResearchAnswer(answer="Claim [1].", cited_source_ids=[1, 1, 1])
        validation = tm.validate_citations(answer, registry)
        result = tm.RunResult(
            question="q?",
            model="m",
            answer=answer,
            registry=registry,
            validation=validation,
            tool_calls=[{"name": "tavily_search", "args": {}}],
            latency_seconds=1.23456,
            run_id="abc",
        )
        record = result.to_log_record()
        assert record["question"] == "q?"
        assert record["cited_source_ids"] == [1]  # deduplicated
        assert record["num_sources"] == 1
        assert record["latency_seconds"] == 1.235
        assert record["langsmith_run_id"] == "abc"
        assert record["validation"]["valid"] is True


# --------------------------------------------------------------------------------------
# CLI: env-var guard rails (no agent invocation needed)
# --------------------------------------------------------------------------------------

class TestCliEnvGuards:
    def test_missing_tavily_key_exits_nonzero(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = runner.invoke(tm.app, ["what is up"])
        assert result.exit_code == 1
        assert "TAVILY_API_KEY" in result.output

    def test_missing_nebius_key_exits_nonzero(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        result = runner.invoke(tm.app, ["what is up"])
        assert result.exit_code == 1
        assert "NEBIUS_API_KEY" in result.output


# --------------------------------------------------------------------------------------
# CLI rendering helpers
# --------------------------------------------------------------------------------------

class TestFormatToolResult:
    def test_formats_tavily_style_payload(self):
        payload = json.dumps(
            {
                "query": "capital of France",
                "results": [{"id": 1, "title": "Wiki", "url": "https://a.com", "content": "Paris"}],
            }
        )
        formatted = tm.format_tool_result(payload)
        assert "[1] Wiki" in formatted
        assert "https://a.com" in formatted

    def test_notes_deduped_count_when_present(self):
        payload = json.dumps({"query": "q", "results": [], "duplicate_results_removed": 2})
        formatted = tm.format_tool_result(payload)
        assert "deduped 2" in formatted

    def test_non_json_string_falls_back_to_truncate(self):
        assert tm.format_tool_result("plain text, not json") == "plain text, not json"

    def test_dict_without_results_key_falls_back_to_truncate(self):
        assert tm.format_tool_result({"foo": "bar"}) == str({"foo": "bar"})


class TestTruncate:
    def test_short_text_unchanged(self):
        assert tm.truncate("short") == "short"

    def test_long_text_is_truncated_with_ellipsis(self):
        text = "x" * 1000
        truncated = tm.truncate(text, limit=50)
        assert len(truncated) <= 53
        assert truncated.endswith("...")

    def test_exact_limit_is_not_truncated(self):
        text = "x" * 50
        assert tm.truncate(text, limit=50) == text


# --------------------------------------------------------------------------------------
# End-to-end: full agent graph wiring with a scripted fake model (no network)
# --------------------------------------------------------------------------------------

class FakeToolCallingChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel can't be bound with tools by default
    (BaseChatModel.bind_tools raises NotImplementedError). Override it to a no-op so
    create_agent's ReAct graph can run against scripted, deterministic responses."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _patch_raw_results(monkeypatch, tool, raw_results: dict) -> None:
    """TavilySearchAPIWrapper is a pydantic model with no `raw_results` field, so it
    rejects instance-attribute assignment. Patch the method on its class instead
    (monkeypatch auto-restores it after the test)."""
    monkeypatch.setattr(type(tool.api_wrapper), "raw_results", lambda self, **kwargs: raw_results)


class TestAgentEndToEnd:
    """Drives the real create_agent graph end-to-end against a scripted model and a
    monkeypatched Tavily call -- no network. Proves SourceRegistry, response_format,
    and validate_citations are wired together correctly, not just individually correct.
    """

    def build_scripted_agent(self, monkeypatch):
        search_tool = tm.TavilySearchWithRegistry()
        _patch_raw_results(
            monkeypatch,
            search_tool,
            {
                "query": "capital of France",
                "results": [
                    {
                        "title": "Paris - Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Paris",
                        "content": "Paris is the capital of France.",
                    },
                ],
            },
        )

        search_call = AIMessage(
            content="",
            tool_calls=[
                {"name": "tavily_search", "args": {"query": "capital of France"}, "id": "call_1"}
            ],
        )
        structured_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchAnswer",
                    "args": {"answer": "The capital of France is Paris [1].", "cited_source_ids": [1]},
                    "id": "call_2",
                }
            ],
        )
        chat_model = FakeToolCallingChatModel(responses=[search_call, structured_call])

        from langchain.agents import create_agent

        agent = create_agent(
            model=chat_model,
            tools=[search_tool],
            system_prompt=tm.SYSTEM_PROMPT,
            response_format=tm.ResearchAnswer,
        )
        return agent, search_tool

    def test_full_graph_produces_validated_structured_answer(self, monkeypatch):
        agent, search_tool = self.build_scripted_agent(monkeypatch)
        state = agent.invoke({"messages": [{"role": "user", "content": "What is the capital of France?"}]})

        structured = state["structured_response"]
        assert isinstance(structured, tm.ResearchAnswer)
        assert structured.cited_source_ids == [1]

        validation = tm.validate_citations(structured, search_tool.registry)
        assert validation.valid

    def test_registry_reflects_the_tool_call_made_during_the_run(self, monkeypatch):
        agent, search_tool = self.build_scripted_agent(monkeypatch)
        agent.invoke({"messages": [{"role": "user", "content": "What is the capital of France?"}]})
        assert len(search_tool.registry) == 1
        assert search_tool.registry.get(1).url == "https://en.wikipedia.org/wiki/Paris"

    def test_run_query_helper_wraps_the_same_graph(self, monkeypatch):
        agent, search_tool = self.build_scripted_agent(monkeypatch)
        result = tm.run_query(
            "What is the capital of France?", agent=agent, search_tool=search_tool, log=False
        )
        assert result.validation.valid
        assert result.answer.cited_source_ids == [1]
        assert len(result.registry) == 1
        assert result.tool_calls[0]["name"] == "tavily_search"

    def test_run_query_detects_hallucinated_citation(self, monkeypatch):
        """If the model cites an id the registry never saw, validation must catch it --
        this is the core guarantee the whole system exists to provide."""
        search_tool = tm.TavilySearchWithRegistry()
        _patch_raw_results(
            monkeypatch,
            search_tool,
            {
                "query": "q",
                "results": [{"title": "Real Source", "url": "https://real.com", "content": "..."}],
            },
        )
        search_call = AIMessage(
            content="",
            tool_calls=[{"name": "tavily_search", "args": {"query": "q"}, "id": "call_1"}],
        )
        hallucinated_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchAnswer",
                    "args": {"answer": "A fabricated claim [1][2].", "cited_source_ids": [1, 2]},
                    "id": "call_2",
                }
            ],
        )
        chat_model = FakeToolCallingChatModel(responses=[search_call, hallucinated_call])

        from langchain.agents import create_agent

        agent = create_agent(
            model=chat_model,
            tools=[search_tool],
            system_prompt=tm.SYSTEM_PROMPT,
            response_format=tm.ResearchAnswer,
        )
        result = tm.run_query("q", agent=agent, search_tool=search_tool, log=False)
        assert not result.validation.valid
        assert 2 in result.validation.missing_ids
