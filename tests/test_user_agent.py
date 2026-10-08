"""The User-Agent names this version and where to read about the tool (2.5.8).

2.5.7 still sent "rolescan/2.2 (personal job search tool)": three releases
stale, and no way for a site owner to find out what was calling."""

from __future__ import annotations

import httpx
import respx

from rolescan import __version__
from rolescan.config import Config, HTTPConfig, default_user_agent
from rolescan.http import Fetcher

PROJECT = "https://github.com/younis-y/deterministic-llm-pipeline"


def test_the_default_names_this_version_and_the_project() -> None:
    assert HTTPConfig().user_agent == f"rolescan/{__version__} (+{PROJECT})"
    assert default_user_agent() == HTTPConfig().user_agent


def test_a_contact_url_in_the_config_replaces_the_projects() -> None:
    cfg = Config.model_validate({"http": {"contact_url": "https://example.org/me"}})
    assert cfg.http.user_agent == f"rolescan/{__version__} (+https://example.org/me)"


def test_an_explicit_user_agent_is_sent_as_it_is() -> None:
    cfg = HTTPConfig(user_agent="my-scanner/1.0", contact_url="https://example.org")
    assert cfg.user_agent == "my-scanner/1.0"


@respx.mock
async def test_the_fetcher_sends_it() -> None:
    route = respx.get("https://boards.example/api").mock(
        return_value=httpx.Response(200, json={})
    )
    async with Fetcher(HTTPConfig()) as f:
        await f.fetch_json("https://boards.example/api")
    sent = route.calls.last.request.headers["User-Agent"]
    assert sent == f"rolescan/{__version__} (+{PROJECT})"
