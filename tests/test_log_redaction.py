"""A key that travels in a URL stays out of the log lines that show the URL.

Jooble's API puts the key in the request path. The source never puts it in an
error or a note, but a run with `--verbose` has httpx log every request URL
and the shared fetcher log the URL it retries, so the key reached the console
and anything that kept it. A filter on those two loggers replaces every key a
source has registered.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

import httpx
import pytest
import respx

from rolescan.config import HTTPConfig, SourceEntry
from rolescan.http import REDACTED, Fetcher, RedactSecrets, hide_in_logs
from rolescan.sources import get_source

KEY = "k3y-0000-example"
JOOBLE = f"https://jooble.org/api/{KEY}"


def _record(message: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, message, args, None)


def _filtered(redactor: RedactSecrets, message: str, *args: object) -> str:
    record = _record(message, *args)
    assert redactor.filter(record) is True, "a filter here never drops a line"
    return record.getMessage()


# --- the filter ----------------------------------------------------------------


def test_a_registered_secret_is_replaced_wherever_it_appears() -> None:
    redactor = RedactSecrets()
    redactor.add(KEY)

    assert _filtered(redactor, f"POST {JOOBLE} and {KEY}") == (
        f"POST https://jooble.org/api/{REDACTED} and {REDACTED}"
    )


def test_a_secret_given_as_a_log_argument_is_replaced_too() -> None:
    redactor = RedactSecrets()
    redactor.add(KEY)

    assert _filtered(
        redactor, "retrying %s in %.1fs (%s)", JOOBLE, 1.5, "HTTP 503"
    ) == (f"retrying https://jooble.org/api/{REDACTED} in 1.5s (HTTP 503)")


def test_the_percent_encoded_form_is_replaced_too() -> None:
    redactor = RedactSecrets()
    secret = "ab/cd ef+gh"
    redactor.add(secret)

    line = f"POST https://jooble.org/api/{quote(secret, safe='')}"

    assert secret not in _filtered(redactor, line)
    assert quote(secret, safe="") not in _filtered(redactor, line)


def test_a_line_without_a_secret_is_untouched_and_nothing_registered_is_a_no_op() -> (
    None
):
    redactor = RedactSecrets()
    assert _filtered(redactor, "nothing here: %s", KEY) == f"nothing here: {KEY}"
    redactor.add(KEY)
    assert (
        _filtered(redactor, "GET https://example.test/") == "GET https://example.test/"
    )


@pytest.mark.parametrize("empty", ["", "   ", "ab"])
def test_an_empty_or_tiny_secret_is_ignored_not_made_into_a_wildcard(
    empty: str,
) -> None:
    redactor = RedactSecrets()
    redactor.add(empty)

    assert _filtered(redactor, "abab banana") == "abab banana"


def test_the_longest_secret_goes_first() -> None:
    redactor = RedactSecrets()
    redactor.add("secret-one")
    redactor.add("secret-one-and-more")

    assert _filtered(redactor, "x secret-one-and-more y") == f"x {REDACTED} y"


# --- what a Jooble scan logs ---------------------------------------------------


def _jooble() -> SourceEntry:
    return SourceEntry(
        kind="jooble", slug="gb", label="Jooble", api_key=KEY, queries=["analyst"]
    )


@respx.mock
async def test_a_verbose_jooble_read_logs_no_key(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request line httpx writes at INFO and the retry line the fetcher
    writes at DEBUG both carry the URL, and so the key."""

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("rolescan.http.asyncio.sleep", no_sleep)
    answers = iter(
        [httpx.Response(503), httpx.Response(200, json={"totalCount": 0, "jobs": []})]
    )
    respx.post(JOOBLE).mock(side_effect=lambda request: next(answers))
    caplog.set_level(logging.DEBUG)
    # A CLI test earlier in the run may have raised httpx's level to WARNING.
    caplog.set_level(logging.DEBUG, logger="httpx")

    async with Fetcher(HTTPConfig(max_retries=1)) as fetcher:
        await get_source(_jooble(), fetcher).fetch()

    assert "HTTP Request" in caplog.text, "httpx's own line was written"
    assert "retrying" in caplog.text, "so was the fetcher's"
    assert KEY not in caplog.text
    assert f"https://jooble.org/api/{REDACTED}" in caplog.text


def test_registering_a_key_twice_is_harmless() -> None:
    hide_in_logs(KEY)
    hide_in_logs(KEY)
    hide_in_logs("")


# --- what an Adzuna scan logs (2.6.0) -------------------------------------------

ADZUNA_ID = "adzuna-id-example"
ADZUNA_KEY = "adzuna-key-example"


@respx.mock
async def test_a_verbose_adzuna_read_logs_neither_credential(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Adzuna's app_id and app_key travel as query parameters, so the request
    line httpx writes carries both unless they are registered."""
    respx.get(url__startswith="https://api.adzuna.com/").mock(
        return_value=httpx.Response(200, json={"count": 0, "results": []})
    )
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="httpx")
    entry = SourceEntry(
        kind="adzuna",
        slug="gb",
        label="Adzuna",
        app_id=ADZUNA_ID,
        app_key=ADZUNA_KEY,
        queries=["analyst"],
    )

    async with Fetcher(HTTPConfig(max_retries=1)) as fetcher:
        await get_source(entry, fetcher).fetch()

    assert "HTTP Request" in caplog.text, "httpx's request line was written"
    assert ADZUNA_ID not in caplog.text
    assert ADZUNA_KEY not in caplog.text
