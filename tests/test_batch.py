"""`llm.batch`: facts calls go through the Message Batches API (2.9.0).

A scan nobody is waiting on can pay the batch price, half the standard one.
The scorer is unchanged in what it decides: the same requests are sent, in one
batch instead of one after another, and every answer is read the same way.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from rolescan.config import LLMConfig, ProfileConfig
from rolescan.models import FitVerdict, Job, ScoredJob
from rolescan.scoring import FitScorer
from rolescan.scoring.enrich import _REGISTRY as _ENRICHERS
from rolescan.scoring.enrich import Enricher, register_enricher
from rolescan.scoring.facts import (
    FieldFact,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
)
from rolescan.scoring.judges import AnthropicJudge


def _facts() -> PostingFacts:
    return PostingFacts(
        level=LevelFact(),
        years_required=YearsFact(),
        student_only=StudentFact(),
        hard_bars=[],
        field=FieldFact(),
        fit_score=70,
        reason="Matches the core skills.",
        keywords_missing=[],
    )


def _message(text: str | None = None, stop_reason: str = "end_turn") -> SimpleNamespace:
    body = _facts().model_dump_json() if text is None else text
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=body)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=700,
            cache_read_input_tokens=10_000,
            cache_creation_input_tokens=0,
            output_tokens=180,
        ),
    )


class _FakeBatches:
    """Stands in for `AsyncAnthropic().messages.batches`.

    Every method takes the keywords the SDK's takes and no others. `polls` is
    how many `retrieve` calls answer "in_progress" before the batch ends;
    `outcomes` maps a request's position in the batch to a result type other
    than "succeeded".
    """

    def __init__(
        self,
        *,
        polls: int = 1,
        outcomes: dict[int, str] | None = None,
        create_error: Exception | None = None,
        never_ends: bool = False,
    ) -> None:
        self.created: list[list[dict[str, Any]]] = []
        self.retrieved = 0
        self.cancelled: list[str] = []
        self.polls = polls
        self.outcomes = outcomes or {}
        self.create_error = create_error
        self.never_ends = never_ends

    async def create(self, *, requests: Any) -> SimpleNamespace:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(list(requests))
        return SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    async def retrieve(self, message_batch_id: str) -> SimpleNamespace:
        self.retrieved += 1
        ended = self.cancelled or (not self.never_ends and self.retrieved > self.polls)
        return SimpleNamespace(
            id=message_batch_id,
            processing_status="ended" if ended else "in_progress",
        )

    async def cancel(self, message_batch_id: str) -> SimpleNamespace:
        self.cancelled.append(message_batch_id)
        return SimpleNamespace(id=message_batch_id, processing_status="canceling")

    async def results(self, message_batch_id: str) -> Any:
        requests = self.created[-1]
        outcomes = self.outcomes
        cancelled = bool(self.cancelled)

        async def rows() -> Any:
            for i, request in enumerate(requests):
                kind = "canceled" if cancelled else outcomes.get(i, "succeeded")
                if kind == "succeeded":
                    result = SimpleNamespace(type=kind, message=_message())
                elif kind == "truncated":
                    result = SimpleNamespace(
                        type="succeeded",
                        message=_message('{"level":', stop_reason="max_tokens"),
                    )
                elif kind == "errored":
                    error = SimpleNamespace(type="invalid_request", message="bad")
                    result = SimpleNamespace(
                        type=kind, error=SimpleNamespace(error=error)
                    )
                else:
                    result = SimpleNamespace(type=kind)
                yield SimpleNamespace(custom_id=request["custom_id"], result=result)

        return rows()


class _FakeMessages:
    """`AsyncAnthropic().messages`: the synchronous `parse` and `batches`."""

    def __init__(self, batches: _FakeBatches, *, parse_error: bool = False) -> None:
        self.batches = batches
        self.parsed: list[dict[str, Any]] = []
        self.parse_error = parse_error

    async def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: Any,
        system: Any = None,
        output_format: Any = None,
        tools: Any = None,
        extra_body: Any = None,
    ) -> SimpleNamespace:
        self.parsed.append(
            {"system": system, "messages": messages, "extra_body": extra_body}
        )
        if self.parse_error:
            raise RuntimeError("api down")
        return SimpleNamespace(parsed_output=_facts(), stop_reason="end_turn")


async def _no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


def _judge(
    batches: _FakeBatches, *, parse_error: bool = False, **llm: Any
) -> tuple[AnthropicJudge, _FakeMessages]:
    llm.setdefault("batch", True)
    judge = AnthropicJudge(LLMConfig(backend="anthropic", api_key="k", **llm))
    messages = _FakeMessages(batches, parse_error=parse_error)
    judge._client = SimpleNamespace(messages=messages)
    judge._sleep = _no_sleep
    return judge, messages


async def _ask(judge: AnthropicJudge, n: int) -> list[Any]:
    return await asyncio.gather(
        *(judge.facts("sys", f"user {i}") for i in range(n)),
        return_exceptions=True,
    )


# --- the judge -------------------------------------------------------------


def test_batching_is_off_unless_asked_for() -> None:
    assert LLMConfig().batch is False


async def test_without_batch_every_facts_call_is_its_own_request() -> None:
    batches = _FakeBatches()
    judge, messages = _judge(batches, batch=False)
    out = await _ask(judge, 4)
    assert all(isinstance(f, PostingFacts) for f in out)
    assert len(messages.parsed) == 4
    assert batches.created == []


async def test_concurrent_facts_calls_go_out_as_one_batch() -> None:
    batches = _FakeBatches()
    judge, messages = _judge(batches)
    out = await _ask(judge, 5)
    assert all(isinstance(f, PostingFacts) for f in out)
    # One synchronous call writes the shared prompt to the cache; the other
    # four ride the batch and read it.
    assert len(messages.parsed) == 1
    assert len(batches.created) == 1
    assert len(batches.created[0]) == 4
    assert len({r["custom_id"] for r in batches.created[0]}) == 4
    sent = {r["params"]["messages"][0]["content"] for r in batches.created[0]}
    assert sent == {f"user {i}" for i in range(1, 5)}


async def test_the_batch_asks_for_the_same_structured_answer() -> None:
    batches = _FakeBatches()
    judge, _ = _judge(batches, model="claude-haiku-4-5", max_tokens=900)
    await _ask(judge, 4)
    params = batches.created[0][0]["params"]
    assert params["model"] == "claude-haiku-4-5"
    assert params["max_tokens"] == 900
    assert params["temperature"] == 0.0
    fmt = params["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "fit_score" in fmt["schema"]["properties"]


async def test_the_warming_call_and_the_batch_share_a_one_hour_cache_entry() -> None:
    """A batch can start after the five-minute entry has expired, and a miss
    on every request costs more than the batch discount saves."""
    batches = _FakeBatches()
    judge, messages = _judge(batches)
    await _ask(judge, 4)
    hour = {"type": "ephemeral", "ttl": "1h"}
    assert messages.parsed[0]["system"][0]["cache_control"] == hour
    for request in batches.created[0]:
        assert request["params"]["system"] == messages.parsed[0]["system"]


async def test_a_model_that_refused_temperature_is_batched_without_it() -> None:
    batches = _FakeBatches()
    judge, _ = _judge(batches)
    judge._send_temperature = False
    await _ask(judge, 4)
    assert "temperature" not in batches.created[0][0]["params"]


async def test_too_few_calls_for_a_batch_are_made_directly() -> None:
    batches = _FakeBatches()
    judge, messages = _judge(batches)
    out = await _ask(judge, 2)
    assert all(isinstance(f, PostingFacts) for f in out)
    assert len(messages.parsed) == 2
    assert batches.created == []


async def test_the_batch_is_polled_until_it_ends() -> None:
    batches = _FakeBatches(polls=3)
    judge, _ = _judge(batches)
    out = await _ask(judge, 4)
    assert all(isinstance(f, PostingFacts) for f in out)
    assert batches.retrieved == 4


async def test_a_request_the_batch_failed_raises_for_that_posting_only() -> None:
    batches = _FakeBatches(outcomes={0: "errored", 1: "expired", 2: "truncated"})
    judge, _ = _judge(batches)
    out = await _ask(judge, 5)
    assert isinstance(out[0], PostingFacts)  # the warming call
    assert isinstance(out[1], RuntimeError) and "invalid_request" in str(out[1])
    assert isinstance(out[2], RuntimeError) and "expired" in str(out[2])
    assert isinstance(out[3], RuntimeError) and "max_tokens" in str(out[3])
    assert isinstance(out[4], PostingFacts)


async def test_a_batch_that_outlives_its_budget_is_cancelled() -> None:
    batches = _FakeBatches(never_ends=True)
    judge, _ = _judge(batches, batch_poll_seconds=30, batch_max_minutes=2)
    out = await _ask(judge, 4)
    assert batches.cancelled == ["msgbatch_1"]
    # Four polls of 30 seconds is the two-minute budget.
    assert isinstance(out[0], PostingFacts)
    assert all(isinstance(e, RuntimeError) and "canceled" in str(e) for e in out[1:])


async def test_a_batch_that_cannot_be_created_falls_back_to_direct_calls() -> None:
    batches = _FakeBatches(create_error=RuntimeError("batches unavailable"))
    judge, messages = _judge(batches)
    out = await _ask(judge, 4)
    assert all(isinstance(f, PostingFacts) for f in out)
    assert len(messages.parsed) == 4


async def test_when_the_warming_call_fails_no_batch_is_sent() -> None:
    """The API is not answering: a batch would only fail later, at a price."""
    batches = _FakeBatches()
    judge, messages = _judge(batches, parse_error=True)
    out = await _ask(judge, 6)
    assert all(isinstance(e, RuntimeError) for e in out)
    assert batches.created == []
    # It stops asking after a few failures in a row rather than trying all six.
    assert len(messages.parsed) < 6


async def test_a_stopped_scan_cancels_the_batch_it_was_waiting_on() -> None:
    """Nothing will read the answers, so the rest of the batch is not paid for."""
    batches = _FakeBatches(never_ends=True)
    judge, _ = _judge(batches, batch_poll_seconds=7, batch_max_minutes=600)
    polling = asyncio.Event()

    async def sleep(seconds: float) -> None:
        if seconds == 7:
            polling.set()
        await asyncio.sleep(0)

    judge._sleep = sleep
    calls = [asyncio.ensure_future(judge.facts("sys", f"u{i}")) for i in range(4)]
    await asyncio.sleep(0)
    flusher = judge._flusher
    assert flusher is not None
    await polling.wait()
    for call in calls[1:]:
        call.cancel()
    flusher.cancel()
    await asyncio.gather(flusher, *calls, return_exceptions=True)
    assert batches.cancelled == ["msgbatch_1"]


async def test_calls_that_arrive_after_a_batch_left_form_the_next_one() -> None:
    batches = _FakeBatches()
    judge, _ = _judge(batches)
    await _ask(judge, 4)
    await _ask(judge, 4)
    assert len(batches.created) == 2


# --- the scorer ------------------------------------------------------------


def _jobs(n: int) -> list[ScoredJob]:
    return [
        ScoredJob(
            job=Job(
                source="test",
                company=f"Acme {i}",
                title="Power Market Analyst",
                location="London",
                url=f"https://x/{i}",
                description=f"Day-ahead price forecasting with Python, desk {i}.",
            ),
            keyword_score=90 - i,
        )
        for i in range(n)
    ]


def _scorer(batches: _FakeBatches, **llm: Any) -> tuple[FitScorer, _FakeMessages]:
    llm.setdefault("batch", True)
    llm.setdefault("max_concurrent", 1)
    cfg = LLMConfig(backend="anthropic", api_key="k", mode="facts", **llm)
    scorer = FitScorer(cfg, ProfileConfig(summary="s", min_report_score=50))
    judge, messages = _judge(batches, **llm)
    scorer._judge = judge
    return scorer, messages


async def test_a_scan_with_one_call_at_a_time_still_sends_one_batch() -> None:
    """`llm.max_concurrent: 1` keeps direct calls in step with the prompt
    cache. It must not hand the batch its requests one at a time."""
    batches = _FakeBatches()
    scorer, messages = _scorer(batches)
    out = await scorer.score_all(_jobs(6))
    assert all(s.fit is not None for s in out)
    assert len(batches.created) == 1
    assert len(batches.created[0]) == 5
    assert len(messages.parsed) == 1
    assert scorer.calls_made == 6


async def test_the_call_ceiling_still_holds_under_batching() -> None:
    batches = _FakeBatches()
    scorer, _ = _scorer(batches, max_calls_per_run=4)
    out = await scorer.score_all(_jobs(7))
    assert sum(1 for s in out if s.fit is not None) == 4
    assert sum(1 for s in out if s.deferred == "llm_ceiling") == 3
    # The ceiling bites on the weakest candidates, as it does without a batch.
    assert {s.job.url for s in out if s.deferred} == {
        f"https://x/{i}" for i in (4, 5, 6)
    }


async def test_a_failed_batch_request_is_a_scoring_error_not_a_lost_scan() -> None:
    batches = _FakeBatches(outcomes={1: "errored"})
    scorer, _ = _scorer(batches)
    out = await scorer.score_all(_jobs(6))
    assert scorer.errors == 1
    assert sum(1 for s in out if s.fit is not None) == 5


class _Tracking(Enricher):
    """Counts how many enrichments run at once."""

    name = "batch-tracking"
    running = 0
    peak = 0
    calls = 0

    async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        cls = type(self)
        cls.running += 1
        cls.calls += 1
        cls.peak = max(cls.peak, cls.running)
        await asyncio.sleep(0)
        cls.running -= 1
        return verdict


async def test_enrichment_stays_one_at_a_time_when_the_facts_were_batched() -> None:
    """A batch answers every posting at once. Enrichment calls are direct and
    share a cached prompt, so they still go out `max_concurrent` at a time."""
    register_enricher(_Tracking)
    try:
        batches = _FakeBatches()
        scorer, _ = _scorer(batches, enricher="batch-tracking")
        await scorer.score_all(_jobs(6))
        assert _Tracking.calls == 6
        assert _Tracking.peak == 1
    finally:
        _ENRICHERS.pop("batch-tracking", None)


async def test_judge_mode_is_not_batched() -> None:
    cfg = LLMConfig(backend="anthropic", api_key="k", mode="judge", batch=True)
    scorer = FitScorer(cfg, ProfileConfig(summary="s", min_report_score=50))
    assert scorer._batching() is False


@pytest.mark.parametrize("field", ["batch_poll_seconds", "batch_max_minutes"])
def test_batch_timings_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        LLMConfig(**{field: 0})
