"""Two-stage scoring: cheap keyword prefilter, then LLM judgement."""

from __future__ import annotations

from rolescan.scoring.cv import CVLibrary, strip_latex
from rolescan.scoring.judges import (
    Judge,
    available_judges,
    get_judge,
    unusable_backend_reason,
)
from rolescan.scoring.keyword import score_keywords
from rolescan.scoring.llm import FitScorer

__all__ = [
    "CVLibrary",
    "FitScorer",
    "Judge",
    "available_judges",
    "get_judge",
    "score_keywords",
    "strip_latex",
    "unusable_backend_reason",
]
