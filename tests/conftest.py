from __future__ import annotations

import json
import re
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from rolescan.config import Config, ProfileConfig
from rolescan.models import Job
from rolescan.pipeline import ScanResult, run_scan
from rolescan.store import Store

FIXTURE_CONFIG = """
profile:
  name: Test Candidate
  summary: A candidate.
  locations: [london, abu dhabi, remote]
  keywords: {energy: 6, data scientist: 7, python: 4, trading: 6, graduate: 4}
  blockers: {uae national: 60, "10+ years": 20, principal: 12}
  hard_blockers: [uae national]
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
    "blockers": [],
    "keywords_missing": ["Kubernetes"],
}


OLLAMA_MODEL = "qwen2.5:7b"
"""A plausible local tag. `LLMConfig.model` defaults to a Claude model, which
is right for the default backend and nonsense for this one, so a test that
mocks ollama sets `llm.model` to this and mocks a server holding it."""


def mock_ollama(
    base_url: str = "http://localhost:11434", model: str = OLLAMA_MODEL
) -> None:
    """Route the ollama backend at a canned verdict, and its preflight at a
    server that is up and has `model` pulled. Call inside @respx.mock.

    Configs under test must set `llm.model` to the same tag, or preflight
    correctly reports the model as not pulled.
    """
    respx.post(f"{base_url}/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": json.dumps(LLM_VERDICT)}}
        )
    )
    respx.get(f"{base_url}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": model}]})
    )


#: Credentials the library reads straight from the environment. Cleared for
#: every test, because a test that only passes on a machine with no keys is a
#: test that passes in CI and fails for the one person actually using the tool.
#: Found the hard way: the Adzuna "no credentials" tests went red the moment
#: real credentials were added to a shell, having been green for weeks.
_CREDENTIAL_VARS = (
    "ADZUNA_APP_ID",
    "ADZUNA_APP_KEY",
    "ANTHROPIC_API_KEY",
    "ROLESCAN_SMTP_PASS",
)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as though the machine has no credentials configured.

    A test that wants one sets it itself; monkeypatch.setenv inside the test
    still wins, because this runs first.
    """
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_REAL_GETADDRINFO = socket.getaddrinfo


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse DNS for anything but localhost, so a test that reaches the real
    network fails here rather than passing on a remote server's answer.

    respx intercepts httpx at the transport, so mocked calls never get this
    far; only an unmocked call does, which is exactly the case to catch."""

    def refuse(host: object, *args: object, **kwargs: object) -> object:
        name = host.decode() if isinstance(host, bytes) else str(host)
        if name in _LOCAL_HOSTS or name == "":
            return _REAL_GETADDRINFO(host, *args, **kwargs)  # type: ignore[arg-type]
        msg = f"test tried to resolve {name!r}; the suite must not reach the network"
        raise socket.gaierror(msg)

    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def profile() -> ProfileConfig:
    return ProfileConfig(
        locations=["london", "abu dhabi", "remote"],
        keywords={"energy": 6, "data scientist": 7, "python": 4, "trading": 6},
        blockers={"uae national": 60, "10+ years": 20},
        hard_blockers=["uae national"],
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


async def scan_and_record(cfg: Config, **kw: Any) -> ScanResult:
    """`run_scan`, then the write the CLI makes once the digest is on disk.

    Since 2.5.7 `run_scan` no longer touches `seen`: it returns what should be
    recorded in `result.to_record` and `rolescan scan` records it after
    `write_digest`. A test that asserts on `seen`, or runs twice and expects
    the second run to skip what the first judged, needs both halves."""
    result = await run_scan(cfg, **kw)
    if result.to_record:
        async with Store(cfg.resolve(cfg.output.db_path)) as store:
            await store.record_all(result.to_record)
    return result


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
