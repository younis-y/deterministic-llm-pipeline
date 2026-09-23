from __future__ import annotations

from pathlib import Path

from rolescan.config import ProfileConfig
from rolescan.models import CVVariant, Job
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
    assert "location mismatch" in score_keywords(job, profile).keyword_penalties


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
