"""`http.bind_interface`: job traffic leaves from a named interface (2.8.0).

A VPN's exit address can be refused by a job site that answers the same request
from the machine's own interface. `http.bind_interface` sends the job-fetching
`Fetcher`'s connections from that interface's IPv4 address, the way
`email.bind_interface` does for mail.

Only the `Fetcher` binds. The model backends' clients must not: a LAN address
as the source would break an Ollama on localhost.

Nothing here reaches a network. respx answers at the transport, and the one
test that opens a real socket opens it to a server on 127.0.0.1.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import FIXTURE_CONFIG, OLLAMA_MODEL, mock_ollama, plain
from rolescan import netif
from rolescan.cli import app
from rolescan.config import Config, HTTPConfig
from rolescan.http import BindError, Fetcher, FetchError
from rolescan.pipeline import run_scan

# RFC 5737 documentation address: nothing here is a real host.
IP_A = "192.0.2.17"

_REAL_CLIENT = httpx.AsyncClient
_REAL_TRANSPORT = httpx.AsyncHTTPTransport


class _Built:
    """What httpx was asked to build, in order."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.transports: list[dict[str, Any]] = []


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> _Built:
    """Record the keywords of every `httpx.AsyncClient` and
    `httpx.AsyncHTTPTransport` built from here on, and build the real thing."""
    record = _Built()

    class Client(_REAL_CLIENT):
        def __init__(self, **kwargs: Any) -> None:
            record.clients.append(kwargs)
            super().__init__(**kwargs)

    class Transport(_REAL_TRANSPORT):
        def __init__(self, **kwargs: Any) -> None:
            record.transports.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", Transport)
    return record


def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def _interface(monkeypatch: pytest.MonkeyPatch, address: str = IP_A) -> list[str]:
    """`interface_ipv4` answers `address`; the list holds each name it was asked."""
    asked: list[str] = []

    def lookup(name: str) -> str:
        asked.append(name)
        return address

    monkeypatch.setattr("rolescan.http.interface_ipv4", lookup)
    return asked


def _no_such_interface(name: str) -> str:
    """What `interface_ipv4` says for an interface that is down or renamed."""
    raise ValueError(
        f"interface {name!r} has no IPv4 address (is it up, and is that still "
        "its name?)"
    )


# --- the config key ----------------------------------------------------------


def test_the_interface_defaults_to_empty() -> None:
    assert HTTPConfig().bind_interface == ""


def test_the_interface_is_read_from_the_config_file() -> None:
    cfg = Config.model_validate({"http": {"bind_interface": "en0"}})
    assert cfg.http.bind_interface == "en0"


# --- unset: the client is built exactly as before ----------------------------


async def test_unset_the_client_is_built_as_it_always_was(
    built: _Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str) -> str:
        raise AssertionError("the interface is read only when one is named")

    monkeypatch.setattr("rolescan.http.interface_ipv4", refuse)
    cfg = HTTPConfig()

    async with Fetcher(cfg):
        pass

    assert built.clients == [
        {
            "timeout": httpx.Timeout(cfg.timeout),
            "headers": {"User-Agent": cfg.user_agent, "Accept": "application/json"},
            "follow_redirects": True,
            "limits": httpx.Limits(
                max_connections=cfg.max_concurrent * 2,
                max_keepalive_connections=cfg.max_concurrent,
            ),
        }
    ]
    assert built.transports == [], "the client builds its own default transport"


# --- set: the transport carries the address ----------------------------------


async def test_set_the_transport_is_bound_and_carries_the_same_limits(
    built: _Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _interface(monkeypatch)
    options: dict[str, Any] = {
        "timeout": 7.5,
        "max_concurrent": 3,
        "user_agent": "ua/1",
    }
    async with Fetcher(HTTPConfig(**options)):
        pass
    unbound_kwargs = built.clients.pop()
    built.transports.clear()
    assert asked == []

    async with Fetcher(HTTPConfig(**options, bind_interface="en0")):
        pass

    (client,) = built.clients
    (transport,) = built.transports
    assert asked == ["en0"]
    limits = httpx.Limits(max_connections=6, max_keepalive_connections=3)
    assert transport == {"local_address": IP_A, "limits": limits}
    # Everything else the client was given is as it was, whatever the interface.
    assert {k: v for k, v in client.items() if k != "transport"} == unbound_kwargs
    assert isinstance(client["transport"], httpx.AsyncHTTPTransport)


async def test_the_address_is_read_once_when_the_fetcher_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _interface(monkeypatch)
    cfg = HTTPConfig(bind_interface="en0")

    with respx.mock:
        route = respx.get("https://boards.example/api").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        async with Fetcher(cfg) as fetcher:
            assert asked == ["en0"], "read when the fetcher opened, before a request"
            for _ in range(3):
                assert await fetcher.fetch_json("https://boards.example/api") == {
                    "ok": True
                }

    assert route.call_count == 3
    assert asked == ["en0"]


@asynccontextmanager
async def _server(body: bytes) -> AsyncIterator[str]:
    """A throwaway HTTP/1.1 server on 127.0.0.1 answering every request with
    `body`. Yields its base URL."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Connection: close\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.close()
        await server.wait_closed()


async def test_the_socket_really_is_bound_to_the_interfaces_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control connects to a local server. With the same server and an
    address this machine does not have as the source, the connect cannot be
    made: the bind is what the connection used, not just what was recorded."""
    _no_proxy_env(monkeypatch)
    _interface(monkeypatch)
    async with _server(b'{"ok": true}') as url:
        async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
            assert await fetcher.fetch_json(url) == {"ok": True}

        async with Fetcher(HTTPConfig(max_retries=0, bind_interface="en0")) as bound:
            with pytest.raises(FetchError, match="ConnectError"):
                await bound.fetch_json(url)


# --- an interface that has no address ----------------------------------------


async def test_an_interface_with_no_ipv4_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rolescan.http.interface_ipv4", _no_such_interface)

    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://boards.example/api").mock(
            return_value=httpx.Response(200, json={})
        )
        fetcher = Fetcher(HTTPConfig(bind_interface="en7"))
        with pytest.raises(
            BindError,
            match=r"^http\.bind_interface: interface 'en7' has no IPv4 address",
        ):
            await fetcher.__aenter__()

        assert route.call_count == 0
    with pytest.raises(RuntimeError, match="async context manager"):
        _ = fetcher.client


async def test_the_real_lookup_error_is_wrapped_with_its_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real `interface_ipv4`, not a stand-in: the wording the email
    path shows for a down interface is the one this path shows."""
    monkeypatch.setattr(netif, "_PLATFORM", "darwin")
    monkeypatch.setattr(netif, "_ifconfig", lambda name: "en7: flags=8822<UP>\n")

    with pytest.raises(BindError) as caught:
        async with Fetcher(HTTPConfig(bind_interface="en7")):
            pass

    assert str(caught.value) == (
        "http.bind_interface: interface 'en7' has no IPv4 address "
        "(is it up, and is that still its name?)"
    )


# --- the commands ------------------------------------------------------------

runner = CliRunner()


def _project(tmp_path: Path, bind: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(FIXTURE_CONFIG + f"http:\n  bind_interface: {bind}\n")
    return path


@respx.mock(assert_all_called=False)
def test_scan_with_a_dead_interface_exits_2_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rolescan.http.interface_ipv4", _no_such_interface)
    board = respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )

    result = runner.invoke(
        app, ["scan", "-c", str(_project(tmp_path, "en7")), "--no-email"]
    )

    out = " ".join(plain(result.output).split())
    assert result.exit_code == 2, out
    assert "http.bind_interface: interface 'en7' has no IPv4 address" in out
    assert "Traceback" not in out
    assert board.call_count == 0
    assert not (tmp_path / "digests").exists(), "no digest for a scan that did not run"


@respx.mock(assert_all_called=False)
def test_discover_with_a_dead_interface_exits_2_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rolescan.http.interface_ipv4", _no_such_interface)
    board = respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )

    result = runner.invoke(app, ["discover", "-c", str(_project(tmp_path, "en7"))])

    out = " ".join(plain(result.output).split())
    assert result.exit_code == 2, out
    assert "http.bind_interface: interface 'en7' has no IPv4 address" in out
    assert "Traceback" not in out
    assert board.call_count == 0


# --- only the job traffic binds ----------------------------------------------


@respx.mock
async def test_the_model_backends_clients_are_not_bound(
    tmp_path: Path, built: _Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan with the interface set and a model on localhost: the Fetcher's
    client is bound, and every client the judges build is the plain one. Binding
    them to a LAN address would break a localhost Ollama."""
    asked = _interface(monkeypatch)
    cfg = Config.model_validate(
        {
            "profile": {
                "keywords": {"energy": 6, "python": 4},
                "min_keyword_score": 0,
                "min_report_score": 10,
            },
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "mode": "judge",
                "model": OLLAMA_MODEL,
                "cascade": False,
            },
            "http": {"bind_interface": "en0"},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Energy Python Analyst",
                        "location": {"name": "London, UK"},
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
                        "content": "energy python",
                    }
                ]
            },
        )
    )
    mock_ollama()

    result = await run_scan(cfg, dry_run=True)

    assert result.llm_calls == 1, "the localhost model answered"
    assert asked == ["en0"], "the interface was read once, for the job fetcher"
    assert [t["local_address"] for t in built.transports] == [IP_A]
    with_transport = [c for c in built.clients if "transport" in c]
    assert len(with_transport) == 1, "only the job fetcher's client is bound"
    judges = [c for c in built.clients if "transport" not in c]
    assert judges, "the model backend built its own clients"
    assert all(set(c) == {"timeout"} for c in judges)


def test_no_judge_module_reads_the_http_bind_setting() -> None:
    source = Path(__file__).parents[1] / "src" / "rolescan" / "scoring"
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "bind_interface" not in text, path.name
        assert "local_address" not in text, path.name
