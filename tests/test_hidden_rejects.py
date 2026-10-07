"""2.5.7: two more ways a role was hidden with no line anywhere.

Weighted `blockers` terms: 82 of 3,357 cached adverts were prefilter rejects
only because of their weight, mostly `head of`, `crypto` and `military`, and
`military` matched the equal-opportunity line "military or veteran status"
on three internships. Near-gate rejects: 20 of 50 good roles in the owner's
labelled set scored under the gate by a few points and were recorded."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from rolescan.config import Config, ProfileConfig
from rolescan.digest import render_markdown
from rolescan.models import Job
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
    """Review focus 3."""
    profile = _profile(title_only_blockers=["head of", "military"])
    body = "Data analyst role. We do not discriminate on military or veteran status."
    in_body = score_keywords(_job("Data Analyst", body), profile)
    assert in_body.keyword_score == 8 * 3 and in_body.keyword_penalties == []
    in_title = score_keywords(_job("Head of Data Analyst", body), profile)
    assert "head of" in in_title.keyword_penalties


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
    assert hidden == [reject]
    text = render_markdown(ScanResult(rule_hidden=hidden))
    assert "Your weighted terms (blockers)" in text and '"military"' in text


def test_a_near_gate_reject_is_listed_under_gate() -> None:
    profile = _profile(
        keywords={"python": 4}, blockers={}, min_keyword_score=20, hidden_gate_margin=10
    )
    reject = score_keywords(_job("Python Analyst", "python"), profile)  # 12 points
    hidden = _rule_hidden([], [], [reject], profile)
    assert hidden == [reject]
    text = render_markdown(ScanResult(rule_hidden=hidden, gate=20))
    assert "Just under your keyword gate" in text and "12 of 20" in text


def test_a_far_under_gate_reject_is_not_listed() -> None:
    profile = _profile(
        keywords={"python": 1}, blockers={}, min_keyword_score=20, hidden_gate_margin=10
    )
    reject = score_keywords(_job("Python Analyst", "python"), profile)  # 3 points
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
