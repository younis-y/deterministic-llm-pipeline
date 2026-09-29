from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from rolescan.config import LLMConfig, ProfileConfig
from rolescan.models import Job, Verdict
from rolescan.scoring import FitScorer, score_keywords
from rolescan.scoring.keyword import TITLE_MULTIPLIER

TEX = r"""
\documentclass[11pt]{article}
\usepackage{charter}
% a comment that must not survive
\begin{document}
{\Huge\textbf{\textcolor{accent}{Ada Lovelace}}}
\section{Experience}
\begin{itemize}
    \item Built production Python pipelines over sensor data.
    \item Cut downtime by 12\% via automated monitoring.
\end{itemize}
\end{document}
"""


def test_title_hits_outweigh_body_hits(profile: ProfileConfig) -> None:
    in_title = Job(
        source="s",
        company="c",
        title="Energy Analyst",
        location="London",
        url="https://x",
        description="",
    )
    in_body = Job(
        source="s",
        company="c",
        title="Analyst",
        location="London",
        url="https://x",
        description="energy",
    )
    a = score_keywords(in_title, profile)
    b = score_keywords(in_body, profile)
    assert a.keyword_score == b.keyword_score * TITLE_MULTIPLIER


def test_gated_role_is_penalised(gated_job: Job, profile: ProfileConfig) -> None:
    scored = score_keywords(gated_job, profile)
    assert "uae national" in scored.keyword_penalties
    assert "uae national" in scored.blocker_hits
    assert scored.keyword_score < profile.min_keyword_score


def test_strong_match_clears_the_prefilter(
    energy_job: Job, profile: ProfileConfig
) -> None:
    scored = score_keywords(energy_job, profile)
    assert scored.keyword_score >= profile.min_keyword_score
    assert not scored.keyword_penalties


def test_wrong_location_is_penalised(profile: ProfileConfig) -> None:
    job = Job(
        source="s",
        company="c",
        title="Energy Data Scientist",
        location="Austin, Texas",
        url="https://x",
    )
    scored = score_keywords(job, profile)
    assert "location mismatch" in scored.keyword_penalties


def test_wrong_location_is_never_a_blocker_hit(profile: ProfileConfig) -> None:
    """A location mismatch lowers the score but is not a hard structural bar,
    so it must never appear in blocker_hits - only real configured blocker
    terms belong there, since blocker_hits is what forces a `blocked`
    verdict."""
    job = Job(
        source="s",
        company="c",
        title="Energy Data Scientist",
        location="Austin, Texas",
        url="https://x",
    )
    assert score_keywords(job, profile).blocker_hits == []


def test_remote_beats_location_mismatch(profile: ProfileConfig) -> None:
    job = Job(
        source="s",
        company="c",
        title="Energy Data Scientist",
        location="Anywhere",
        url="https://x",
        remote=True,
    )
    assert "location mismatch" not in score_keywords(job, profile).keyword_penalties


def test_blank_location_is_not_penalised(profile: ProfileConfig) -> None:
    """Terseness is not evidence of a bad location; let the LLM judge."""
    job = Job(
        source="s",
        company="c",
        title="Energy Data Scientist",
        location="",
        url="https://x",
    )
    assert "location mismatch" not in score_keywords(job, profile).keyword_penalties


def test_empty_profile_scores_zero() -> None:
    job = Job(source="s", company="c", title="Anything", url="https://x")
    assert score_keywords(job, ProfileConfig()).keyword_score == 0


# --- blockers cost points; hard_blockers are walls -------------------------


def _matlab_job(title: str = "Data Scientist") -> Job:
    return Job(
        source="s",
        company="c",
        title=title,
        location="London",
        url="https://x",
        description="Python, energy, trading. Some MATLAB experience useful.",
    )


def test_a_weighted_term_is_penalised_but_not_a_blocker_hit() -> None:
    """matlab: 15 in the real config is a preference ("capable, but does not
    want to use it again"), not a structural bar. It must cost its weight and
    still show up for display, but it must not force `blocked`."""
    profile = ProfileConfig(
        keywords={"energy": 6, "trading": 6},
        blockers={"matlab": 15},
    )
    scored = score_keywords(_matlab_job(), profile)
    assert "matlab" in scored.keyword_penalties
    assert "matlab" not in scored.blocker_hits
    assert scored.keyword_score == 6 + 6 - 15


def _clearance_job() -> Job:
    return Job(
        source="s",
        company="c",
        title="Analyst",
        url="https://x",
        description="energy. Active security clearance required.",
    )


def test_a_hard_blocker_that_is_also_weighted_keeps_its_weight() -> None:
    """The common case: a term that is both expensive and fatal. It must land
    in keyword_penalties (costing its points, as before) AND in blocker_hits
    (forcing `blocked`), from the two independent lists."""
    profile = ProfileConfig(
        keywords={"energy": 6},
        blockers={"security clearance": 60},
        hard_blockers=["security clearance"],
    )
    scored = score_keywords(_clearance_job(), profile)
    assert "security clearance" in scored.keyword_penalties
    assert "security clearance" in scored.blocker_hits
    assert scored.keyword_score == 6 - 60


def test_a_heavy_weight_alone_does_not_block() -> None:
    """The defect this design change exists to kill: weight no longer implies
    hardness. 60 points is a strong preference and nothing more until the term
    is named in hard_blockers, so a weight retune cannot silently create or
    destroy a structural bar."""
    profile = ProfileConfig(keywords={"energy": 6}, blockers={"security clearance": 60})
    scored = score_keywords(_clearance_job(), profile)
    assert "security clearance" in scored.keyword_penalties
    assert scored.blocker_hits == []
    assert not scored.is_blocked


def test_a_hard_blocker_with_no_weight_blocks_and_costs_nothing() -> None:
    """The other configuration the old threshold made unexpressible: a bar
    that is fatal but must not distort prefilter ordering. It blocks at zero
    cost, so unrelated roles are not pushed under min_keyword_score by it."""
    profile = ProfileConfig(keywords={"energy": 6}, hard_blockers=["security clearance"])
    scored = score_keywords(_clearance_job(), profile)
    assert scored.blocker_hits == ["security clearance"]
    assert scored.keyword_penalties == []
    assert scored.keyword_score == 6, "a hard bar need not cost points"
    assert scored.is_blocked


def test_hard_blockers_are_matched_case_and_whitespace_insensitively() -> None:
    """Entries are normalised at load the same way posting text is, so a
    capitalised or double-spaced entry is a working bar rather than a silent
    no-op."""
    profile = ProfileConfig(hard_blockers=["  Security   Clearance "])
    assert profile.hard_blockers == ["security clearance"]
    assert "security clearance" in score_keywords(_clearance_job(), profile).blocker_hits


def test_weighted_term_does_not_block_without_an_llm_verdict() -> None:
    """No-LLM path: a soft blocker still costs its weight and can still push
    the keyword score to (or below) zero, but must not force `blocked` - the
    fallback verdict is decided by the keyword score alone, same as before
    this field existed."""
    profile = ProfileConfig(
        keywords={"energy": 6, "trading": 6}, blockers={"matlab": 15}
    )
    scored = score_keywords(_matlab_job(), profile)
    assert scored.keyword_score == 6 + 6 - 15
    assert scored.verdict is Verdict.SKIP, "keyword score alone decides, not a block"
    assert not scored.is_blocked


def test_hard_blocker_still_blocks_without_an_llm_verdict() -> None:
    """No-LLM path, hard term: unchanged from the previous fix - a real
    blocker still forces `blocked` even with no LLM verdict at all."""
    profile = ProfileConfig(
        keywords={"energy": 6},
        blockers={"security clearance": 60},
        hard_blockers=["security clearance"],
    )
    scored = score_keywords(_clearance_job(), profile)
    assert scored.verdict is Verdict.BLOCKED
    assert scored.is_blocked


# --- hard terms match on word boundaries; weighted terms do not ------------


def _crypto_job(description: str) -> Job:
    return Job(
        source="s",
        company="c",
        title="Quantitative Analyst",
        location="London",
        url="https://x",
        description=description,
    )


def test_a_hard_blocker_does_not_match_inside_a_longer_word() -> None:
    """The live defect: `crypto` as a hard bar also matched "cryptographic",
    so a security-adjacent quant role that mentions cryptographic hashing
    once was blocked, hidden and recorded as seen - an opportunity the reader
    never learns about, on a substring."""
    profile = ProfileConfig(keywords={"energy": 6}, hard_blockers=["crypto"])
    job = _crypto_job("Low-latency systems. Cryptographic hashing and signing.")
    scored = score_keywords(job, profile)
    assert scored.blocker_hits == []
    assert not scored.is_blocked


def test_a_hard_blocker_still_matches_the_whole_word() -> None:
    """The other half: narrowing the match must not stop the bar working."""
    profile = ProfileConfig(keywords={"energy": 6}, hard_blockers=["crypto"])
    scored = score_keywords(_crypto_job("Trading on a crypto exchange."), profile)
    assert scored.blocker_hits == ["crypto"]
    assert scored.is_blocked


def test_a_multi_word_hard_blocker_is_unaffected_by_boundaries() -> None:
    profile = ProfileConfig(hard_blockers=["uae national", "10+ years"])
    job = _crypto_job("Must be a UAE National with 10+ years of experience.")
    assert score_keywords(job, profile).blocker_hits == ["uae national", "10+ years"]


def test_a_weighted_term_does_not_match_inside_a_longer_word() -> None:
    """The asymmetry this replaces was justified by "a wrong points deduction
    is recoverable". It is not, whenever the deduction crosses the prefilter:
    the posting never reaches the LLM, is written to `seen` by the reject
    path, and can never surface again. A 50-point `crypto` charged to
    "cryptographic hashing" deletes a modest role as thoroughly as a hard bar
    would, and more quietly - nothing records that a blocker was involved."""
    profile = ProfileConfig(keywords={"energy": 20}, blockers={"crypto": 15})
    scored = score_keywords(_crypto_job("energy. Cryptographic hashing."), profile)
    assert scored.keyword_penalties == []
    assert scored.keyword_score == 20, "no points come off for a word not present"
    assert scored.blocker_hits == []


def test_a_weighted_term_still_matches_the_whole_word() -> None:
    """The other half: narrowing the match must not stop the weight working."""
    profile = ProfileConfig(keywords={"energy": 20}, blockers={"crypto": 15})
    scored = score_keywords(_crypto_job("energy. Trading on a crypto exchange."), profile)
    assert scored.keyword_penalties == ["crypto"]
    assert scored.keyword_score == 5


def test_a_weighted_seniority_term_does_not_match_a_longer_word() -> None:
    """Both live in the user's config today, and both fire constantly on UK
    and Gulf adverts: `head of` on "head office", `director` on "directorate"
    and on the boilerplate "board of directors"."""
    profile = ProfileConfig(blockers={"head of": 40, "director": 40})
    job = _crypto_job(
        "Based at our head office. Reports to the directorate and to the "
        "board of directors."
    )
    assert score_keywords(job, profile).keyword_penalties == []


def test_a_weighted_seniority_term_matches_the_real_title() -> None:
    profile = ProfileConfig(blockers={"head of": 40, "director": 40})
    job = Job(
        source="s",
        company="c",
        title="Head of Data",
        location="London",
        url="https://x",
        description="You will be a Director of the practice.",
    )
    assert score_keywords(job, profile).keyword_penalties == ["head of", "director"]


# --- a bad hard_blockers list must fail the load, not the digest -----------


def test_an_empty_hard_blocker_is_refused_at_load() -> None:
    """An empty term matches every posting: every role blocked, and with
    show_blocked false, every role deleted. This is the one configuration
    that silently means "everything is hard", so it must not load."""
    with pytest.raises(ValidationError) as e:
        ProfileConfig(hard_blockers=["security clearance", ""])
    assert "hard_blockers[1]" in str(e.value)


def test_a_whitespace_only_hard_blocker_is_refused_at_load() -> None:
    with pytest.raises(ValidationError):
        ProfileConfig(hard_blockers=["   "])


def test_a_single_character_hard_blocker_is_refused_at_load() -> None:
    """Word-boundary matching turns "a" into a bar on the word "a", which
    almost every posting contains."""
    with pytest.raises(ValidationError):
        ProfileConfig(hard_blockers=["a"])


def test_duplicate_hard_blockers_collapse() -> None:
    profile = ProfileConfig(hard_blockers=["UAE National", "uae national"])
    assert profile.hard_blockers == ["uae national"]


def test_a_weighted_term_is_normalised_at_load() -> None:
    """A key the loader leaves alone can never match: the matcher searches
    `Job.blob`, whose whitespace is collapsed, so `security  clearance` with
    two spaces is a weight that is silently never charged."""
    profile = ProfileConfig(blockers={"  Security   Clearance ": 60})
    assert profile.blockers == {"security clearance": 60}
    assert score_keywords(_clearance_job(), profile).keyword_penalties == [
        "security clearance"
    ]


def test_the_two_blocker_fields_cannot_disagree_about_a_term() -> None:
    """The disagreement case: normalising only `hard_blockers` let the same
    term be hard (normalised, matching) and weightless (unnormalised, never
    matching) at once, with nothing reporting it."""
    profile = ProfileConfig(
        blockers={"security  clearance": 60},
        hard_blockers=["security  clearance"],
    )
    scored = score_keywords(_clearance_job(), profile)
    assert scored.blocker_hits == ["security clearance"]
    assert scored.keyword_penalties == ["security clearance"]
    assert scored.keyword_score == -60


def test_two_weighted_keys_that_normalise_to_one_keep_the_heavier_weight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normalisation can merge two keys, and the merged term needs one weight.

    The heavier wins, not the last-written one: "last" is decided by the order
    two lines happen to sit in the YAML, which is invisible to the author and
    would change the prefilter when the file is merely re-sorted. Taking the
    larger of two costs also cannot quietly weaken a bar the author did write.
    """
    with caplog.at_level(logging.WARNING, logger="rolescan.config"):
        profile = ProfileConfig(
            blockers={"Security  Clearance": 20, "security clearance ": 60}
        )
    assert profile.blockers == {"security clearance": 60}
    assert score_keywords(_clearance_job(), profile).keyword_score == -60
    assert "security clearance" in caplog.text, "a silent merge is the bug"


def test_the_heavier_weight_wins_whichever_key_is_written_first() -> None:
    """Order-independence is the whole point of picking by size."""
    a = ProfileConfig(blockers={"Security  Clearance": 60, "security clearance ": 20})
    b = ProfileConfig(blockers={"Security  Clearance": 20, "security clearance ": 60})
    assert a.blockers == b.blockers == {"security clearance": 60}


def test_an_empty_weighted_blocker_is_dropped_at_load() -> None:
    """Not refused, as an empty hard bar is: an empty weight cannot delete
    anything. It is dropped because it is not a term - left in place it took
    a constant off every posting, since `"" in blob` is always true."""
    profile = ProfileConfig(keywords={"energy": 20}, blockers={"   ": 60})
    assert profile.blockers == {}
    assert score_keywords(_crypto_job("energy work"), profile).keyword_score == 20


def test_a_hard_blocker_absent_from_blockers_is_allowed() -> None:
    """Deliberate: a bar that costs nothing is a first-class configuration,
    not a mistake. Rejecting it would force the author to invent a weight for
    something whose whole point is that it does not shape ranking."""
    profile = ProfileConfig(blockers={"matlab": 15}, hard_blockers=["uae national"])
    assert profile.hard_blockers == ["uae national"]


def test_extra_prompt_is_appended_verbatim_and_empty_by_default() -> None:
    """The seam for context this library has no business knowing about.

    A private caller adds its own vocabulary - documents to choose between, a
    house style - without the public prompt learning what any of it means.
    """
    plain = FitScorer(LLMConfig(), ProfileConfig(summary="A candidate."))
    assert "{extra}" not in plain._system()

    private = FitScorer(
        LLMConfig(),
        ProfileConfig(summary="A candidate."),
        extra_prompt="Pick one of: alpha, beta.",
    )
    assert "Pick one of: alpha, beta." in private._system()
    assert len(private._system()) > len(plain._system())


def test_the_public_prompt_names_no_documents() -> None:
    """The judgement is score, verdict, confidence, one sentence and the bars.
    Which CV to send and what to change in it is private, and this asserts the
    public half stays that way."""
    from rolescan.scoring.llm import SYSTEM

    lowered = SYSTEM.lower()
    for word in ("cv", "resume", "tailor", "variant"):
        assert word not in lowered, f"{word!r} leaked back into the public prompt"
