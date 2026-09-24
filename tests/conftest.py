from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
import respx

from rolescan.config import Config, ProfileConfig
from rolescan.models import Job

FIXTURE_CONFIG = """
profile:
  name: Test Candidate
  summary: A candidate.
  locations: [london, abu dhabi, remote]
  keywords: {energy: 6, data scientist: 7, python: 4, trading: 6, graduate: 4}
  blockers: {uae national: 40, "10+ years": 20, principal: 12}
  min_keyword_score: 18
  min_report_score: 55
llm:
  enabled: false
output:
  dir: digests
  db_path: seen.db
sources:
  - {kind: greenhouse, slug: acme, label: Acme}
"""


# The pipeline distinguishes "the intended judge answered" from "the intended
# judge could not start", and records only the first - so a *healthy* scoring
# run has to be expressible in a test, without a key and without a network
# call. Ollama is the backend that needs no key; `mock_ollama` answers for it
# in-process.
LLM_VERDICT = {
    "fit_score": 72,
    "verdict": "consider",
    "confidence": "medium",
    "reason": "Strong power-market overlap, but the role wants five years.",
    "cv_variant": "CV_EnergySystems-Modelling",
    "tailoring": ["Lead with the day-ahead forecasting project."],
    "blockers": [],
    "keywords_missing": ["Kubernetes"],
}


def mock_ollama(
    base_url: str = "http://localhost:11434", model: str = "claude-sonnet-5"
) -> None:
    """Route the ollama backend at a canned verdict, and its preflight at a
    server that is up and has `model` pulled. Call inside @respx.mock.

    `model` defaults to `LLMConfig.model`'s own default, so callers that
    build a config without overriding `llm.model` still pass preflight.
    """
    respx.post(f"{base_url}/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": json.dumps(LLM_VERDICT)}}
        )
    )
    respx.get(f"{base_url}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": model}]})
    )


@pytest.fixture
def profile() -> ProfileConfig:
    return ProfileConfig(
        locations=["london", "abu dhabi", "remote"],
        keywords={"energy": 6, "data scientist": 7, "python": 4, "trading": 6},
        blockers={"uae national": 40, "10+ years": 20},
        min_keyword_score=18,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text(FIXTURE_CONFIG)
    return Config.load(path)


@pytest.fixture
def energy_job() -> Job:
    return Job(
        source="greenhouse",
        company="EDF Trading",
        title="Graduate Data Scientist, Power Markets",
        location="London, United Kingdom",
        url="https://example.com/1",
        description=(
            "Day-ahead electricity price forecasting with Python. Energy "
            "trading desk support, time series modelling."
        ),
        posted="2026-08-20",
    )


@pytest.fixture
def gated_job() -> Job:
    return Job(
        source="smartrecruiters",
        company="Masdar",
        title="Data Scientist",
        location="Abu Dhabi, United Arab Emirates",
        url="https://example.com/2",
        description=(
            "Support analytics and AI product development. Python, SQL, Power "
            "BI. Bachelor's degree; UAE National (National Talent programme). "
            "Exposure to energy and utilities advantageous."
        ),
        posted="2026-08-22",
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """Strip ANSI escapes from rendered CLI output.

    `rich` decides whether to style from the environment, and a shell that
    exports FORCE_COLOR (Claude Code and several CI runners do) makes it emit
    escapes even when the output is captured. Setting no_color is not enough:
    table titles still carry an italic sequence. Assertions on CLI text should
    compare against normalised text rather than depend on the caller's terminal.
    """
    return _ANSI.sub("", text)
