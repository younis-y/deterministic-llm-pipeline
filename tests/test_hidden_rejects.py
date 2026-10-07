"""2.5.7: two more ways a role was hidden with no line anywhere.

Weighted `blockers` terms: 82 of 3,357 cached adverts were prefilter rejects
only because of their weight, mostly `head of`, `crypto` and `military`, and
`military` matched the equal-opportunity line "military or veteran status"
on three internships. Near-gate rejects: 20 of 50 good roles in the owner's
labelled set scored under the gate by a few points and were recorded."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest
import respx

from rolescan.config import Config, ProfileConfig
from rolescan.digest import render_html, render_markdown
from rolescan.models import Job, ScoredJob
from rolescan.pipeline import ScanResult, _prefilter, _rule_hidden, run_scan
from rolescan.scoring.keyword import score_keywords


def _profile(**kw: object) -> ProfileConfig:
    base: dict[str, object] = {
        "keywords": {"data analyst": 8, "python": 4},
        "blockers": {"military": 50, "head of": 20},
        "min_keyword_score": 20,
    }
    base.update(kw)
    return ProfileConfig.model_validate(base)


def _job(title: str, body: str) -> Job:
    return Job(
        source="t",
        company="Acme",
        title=title,
        url="https://acme.example/1",
        description=body,
    )


def test_title_only_term_costs_its_weight_only_in_the_title() -> None:
    """A title-only term is charged its weight in the title and nowhere else."""
    profile = _profile(title_only_blockers=["head of", "military"])
    body = "Data analyst role. We do not discriminate on military or veteran status."
    in_body = score_keywords(_job("Data Analyst", body), profile)
    assert in_body.keyword_score == 8 * 3 and in_body.keyword_penalties == []
    in_title = score_keywords(_job("Head of Data Analyst", body), profile)
    assert "head of" in in_title.keyword_penalties


def test_a_title_only_term_with_no_blockers_weight_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`title_only_blockers` only restricts where a `blockers` weight counts,
    so a term with no weight there does nothing, and used to say nothing."""
    with caplog.at_level(logging.WARNING, logger="rolescan.config"):
        profile = _profile(title_only_blockers=["head of", "Director", "military"])
    inert = [r.message for r in caplog.records if "does nothing" in r.message]
    assert len(inert) == 1 and "'director'" in inert[0]
    assert "Give it a weight in blockers" in inert[0]
    assert profile.title_only_blockers == ["head of", "director", "military"]


def test_a_title_only_term_with_a_weight_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="rolescan.config"):
        _profile(title_only_blockers=["head of", "military"])
    assert "does nothing" not in caplog.text


def test_title_only_terms_are_normalised_like_blockers() -> None:
    """A spelling the loader leaves alone could never name its `blockers` key."""
    profile = _profile(title_only_blockers=["  Head  OF ", "head of", " "])
    assert profile.title_only_blockers == ["head of"]
    assert _profile().title_only_blockers == []
    assert _profile().hidden_gate_margin == 10


def test_a_weighted_only_reject_is_listed_under_blockers() -> None:
    profile = _profile()
    reject = score_keywords(_job("Data Analyst", "military or veteran status"), profile)
    assert reject.keyword_score < profile.min_keyword_score
    hidden = _rule_hidden([], [], [reject], profile)
    assert [(h.job, h.hidden_as) for h in hidden] == [(reject.job, "blockers")]
    text = render_markdown(ScanResult(rule_hidden=hidden))
    assert "Your weighted terms (blockers)" in text and '"military"' in text


def test_a_near_gate_reject_is_listed_under_gate() -> None:
    profile = _profile(
        keywords={"python": 4}, blockers={}, min_keyword_score=20, hidden_gate_margin=10
    )
    reject = score_keywords(_job("Python Analyst", "python"), profile)  # 12 points
    hidden = _rule_hidden([], [], [reject], profile)
    assert [(h.job, h.hidden_as) for h in hidden] == [(reject.job, "gate")]
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=20))
    assert "Just under your keyword gate" in text and "12 of 20" in text


def test_a_far_under_gate_reject_is_not_listed() -> None:
    profile = _profile(
        keywords={"python": 1}, blockers={}, min_keyword_score=20, hidden_gate_margin=10
    )
    reject = score_keywords(_job("Python Analyst", "python"), profile)  # 3 points
    assert _rule_hidden([], [], [reject], profile) == []


def test_a_small_penalty_that_could_not_have_cleared_the_gate_is_not_blamed() -> None:
    """Gate 20, 19 without a 5-point `crypto`: still under, so "pushed under
    the gate by crypto" would be false. The group is decided where the weights
    are (`_rule_hidden`), not guessed from the penalty list."""
    profile = _profile(keywords={"python": 4, "sql": 7}, blockers={"crypto": 5})
    reject = score_keywords(_job("Python Analyst", "sql, crypto"), profile)
    assert reject.keyword_score == 14 and reject.keyword_penalties == ["crypto"]
    hidden = _rule_hidden([], [], [reject], profile)
    assert [h.hidden_as for h in hidden] == ["gate"]
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=20))
    assert "Just under your keyword gate" in text and "scored 14 of 20" in text
    assert "Your weighted terms (blockers)" not in text
    assert "pushed under" not in text


def test_a_reject_both_weighted_and_near_the_gate_is_listed_once_under_blockers() -> (
    None
):
    """Score 15 with a 10-point term: 25 without it, so the term did it, and
    15 is also inside the margin. The `listed` check keeps it out of `near`."""
    profile = _profile(
        keywords={"data analyst": 8, "python": 1}, blockers={"head of": 10}
    )
    reject = score_keywords(_job("Head of Data Analyst", "python"), profile)
    assert reject.keyword_score == 15 and reject.keyword_penalties == ["head of"]
    hidden = _rule_hidden([], [], [reject], profile)
    assert [h.hidden_as for h in hidden] == ["blockers"]
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=20))
    assert text.count("Head of Data Analyst") == 1
    assert "Just under your keyword gate" not in text


def test_the_html_digest_lists_both_new_groups() -> None:
    profile = _profile(blockers={"military": 50})
    weighted = score_keywords(
        _job("Data Analyst", "military or veteran status"), profile
    )
    near = score_keywords(_job("Python Analyst", "python"), profile)  # 12 points
    hidden = _rule_hidden([], [], [weighted, near], profile)
    assert [h.hidden_as for h in hidden] == ["blockers", "gate"]
    html = render_html(ScanResult(rule_hidden=hidden, gate=20))
    assert "Your weighted terms (blockers)" in html
    assert "Just under your keyword gate" in html
    assert "pushed under the gate by" in html and "scored 12 of 20" in html


def test_a_title_only_term_found_only_in_the_body_gives_no_weight_back() -> None:
    """`military` is weighted, title-only AND a hard blocker. In the body it
    bars but costs nothing, so a reject that was low relevance to begin with
    is not "pushed under" by a weight it was never charged."""
    profile = _profile(
        keywords={"data analyst": 8, "python": 4},
        blockers={"military": 50},
        hard_blockers=["military"],
        title_only_blockers=["military"],
    )
    body = "python. military or veteran status."
    in_body = score_keywords(_job("Intern", body), profile)
    assert in_body.blocker_hits == ["military"] and in_body.keyword_penalties == []
    assert in_body.keyword_score == 4
    in_title = score_keywords(_job("Military Data Analyst", ""), profile)
    assert in_title.blocker_hits == ["military"]
    assert in_title.keyword_penalties == ["military"]
    assert in_title.keyword_score == 24 - 50
    hidden = _rule_hidden([], [], [in_body, in_title], profile)
    assert [h.job.title for h in hidden] == ["Military Data Analyst"]
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=20))
    assert 'blocked by "military"' in text


def test_a_reject_with_an_unweighted_hit_and_weighted_penalties_is_listed_once() -> (
    None
):
    """Score -11 after `director` (40) and `10+ years` (20); 49 without them.
    The `hard_blockers` hit `security clearance` has no `blockers` weight, so
    it is not what was charged. Listed nowhere before: `pushed_under` counted
    only the hit's own charge and `weighted` required no hits at all. Now
    either kind of push lists it, once, under the terms group."""
    profile = _profile(
        keywords={"python": 4},
        blockers={"director": 40, "10+ years": 20},
        hard_blockers=["security clearance"],
        min_keyword_score=18,
    )
    reject = ScoredJob(
        job=_job("Director of Python", "10+ years, security clearance"),
        keyword_score=-11,
        keyword_penalties=["director", "10+ years"],
        blocker_hits=["security clearance"],
    )
    hidden = _rule_hidden([], [], [reject], profile)
    assert [h.job for h in hidden] == [reject.job], "listed once"
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=18))
    assert text.count("Director of Python") == 1
    assert "Your blocking terms" in text and 'blocked by "security clearance"' in text


def test_an_unweighted_hit_with_no_push_is_still_not_listed() -> None:
    """The fix lists a reject only when the weights did put it under the gate:
    a far-under reject with an unweighted hit and nothing charged stays out."""
    profile = _profile(
        keywords={"python": 4},
        blockers={},
        hard_blockers=["security clearance"],
        min_keyword_score=18,
        hidden_gate_margin=0,
    )
    reject = ScoredJob(
        job=_job("Analyst", "security clearance"),
        keyword_score=0,
        blocker_hits=["security clearance"],
    )
    assert _rule_hidden([], [], [reject], profile) == []


def test_a_thin_posting_a_hard_blocker_term_caught_is_listed_but_no_other_thin_is() -> (
    None
):
    """A posting with no description is deferred, not rejected, so it is no
    longer in `rejects`; without `thin=` a title that matched `hard_blockers`
    would be in no list at all. Only that one is listed: a thin near-gate or
    weighted reject would repeat every run until its text arrives, and so
    would one whose only hit is an excluded location, which the reader has
    nothing to check."""
    profile = _profile(
        keywords={"python": 4},
        blockers={},
        hard_blockers=["security clearance"],
        excluded_locations=["dubai"],
        hidden_gate_margin=10,
    )
    caught = score_keywords(_job("Python Analyst, security clearance", ""), profile)
    plain = score_keywords(_job("Python Analyst", ""), profile)  # 12: near the gate
    dubai = score_keywords(
        _job("Python Analyst", "").model_copy(
            update={"location": "Dubai", "url": "https://acme.example/2"}
        ),
        profile,
    )
    assert dubai.blocker_hits == ["location: dubai"]
    _, rejects, thin = _prefilter([caught, plain, dubai], profile.min_keyword_score)
    assert rejects == [] and [s.deferred for s in thin] == ["thin"] * 3
    assert _rule_hidden([], [], rejects, profile) == [], "the default lists none"
    hidden = _rule_hidden([], [], rejects, profile, thin=thin)
    assert [s.job.title for s in hidden] == ["Python Analyst, security clearance"]
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=20))
    assert 'blocked by "security clearance"' in text
    assert "Just under your keyword gate" not in text


@respx.mock
async def test_run_scan_carries_gate_and_the_thin_posting_a_term_caught(
    tmp_path: Path,
) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": jid,
                        "title": title,
                        "location": {"name": "London"},
                        "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                        "content": "",
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                    for jid, title in (
                        (1, "Python Analyst, security clearance"),
                        (2, "Python Analyst"),
                    )
                ]
            },
        )
    )
    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": {"python": 4},
                "hard_blockers": ["security clearance"],
                "min_keyword_score": 20,
            },
            "llm": {"enabled": False},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )

    result = await run_scan(cfg, dry_run=True, check_llm=False)

    assert result.gate == 20
    assert sorted(s.job.title for s in result.deferred) == [
        "Python Analyst",
        "Python Analyst, security clearance",
    ]
    assert [s.job.title for s in result.rule_hidden] == [
        "Python Analyst, security clearance"
    ]
