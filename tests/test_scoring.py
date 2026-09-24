from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rolescan.config import ProfileConfig
from rolescan.models import CVVariant, Job, Verdict
from rolescan.scoring import score_keywords
from rolescan.scoring.cv import CVLibrary, strip_latex
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


# --- CV library ------------------------------------------------------------


def test_strip_latex_keeps_words_drops_markup() -> None:
    out = strip_latex(TEX)
    assert "Ada Lovelace" in out
    assert "Built production Python pipelines" in out
    assert "comment that must not survive" not in out
    assert "\\documentclass" not in out
    assert "\\begin" not in out
    assert "{" not in out and "}" not in out
    assert "charter" not in out, "preamble should be dropped"


def test_cv_library_loads_variants(tmp_path: Path) -> None:
    (tmp_path / "CV_EnergySystems-Modelling.tex").write_text(TEX)
    (tmp_path / "CV_Quant-Trading.md").write_text("# Quant CV\nVECM, futures.")
    library = CVLibrary.load(tmp_path)
    assert len(library) == 2
    assert CVVariant.ENERGY in library.variants
    assert "VECM" in library.variants[CVVariant.QUANT]
    block = library.prompt_block()
    assert '<cv name="CV_EnergySystems-Modelling">' in block


def test_missing_cv_dir_degrades_gracefully(tmp_path: Path) -> None:
    library = CVLibrary.load(tmp_path / "nope")
    assert not library
    assert "not available" in library.prompt_block()
    assert CVVariant.ENERGY.value in library.prompt_block()


def test_cv_library_handles_none() -> None:
    assert not CVLibrary.load(None)


# --- CV filenames do not always match the enum spelling --------------------


def test_cv_variant_matches_a_punctuated_filename(tmp_path: Path) -> None:
    """The enum value may differ in punctuation from the filename; folding
    normalizes both. cv_ai_llm_engineering.tex matches CV_AI-LLM-Engineering."""
    d = tmp_path / "cvs"
    d.mkdir()
    (d / "cv_ai_llm_engineering.tex").write_text(
        "\\begin{document}Machine learning and AI CV\\end{document}", encoding="utf-8"
    )
    (d / "CV_Energy_Systems_Modelling.tex").write_text(
        "\\begin{document}Energy markets CV\\end{document}", encoding="utf-8"
    )
    library = CVLibrary.load(d)
    assert CVVariant.AI_LLM in library.variants, "cv_ai_llm_engineering.tex must map to CV_AI-LLM-Engineering"
    assert CVVariant.ENERGY in library.variants
    assert len(library) == 2


def test_cv_loading_is_case_and_separator_insensitive(tmp_path: Path) -> None:
    d = tmp_path / "cvs"
    d.mkdir()
    (d / "cv_data_science_gulf.tex").write_text(
        "\\begin{document}Data\\end{document}", encoding="utf-8"
    )
    assert CVVariant.DATA_SCIENCE in CVLibrary.load(d).variants
