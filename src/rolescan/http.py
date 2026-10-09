"""Shared async HTTP client.

One pooled client for the whole run, a semaphore so a big employer list cannot
open eighty sockets at once, and retries that back off with jitter and honour
Retry-After. Sources get a plain `fetch_json` and never touch any of this.
"""

from __future__ import annotations

import asyncio
import logging
import random
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote

import httpx

from rolescan.config import HTTPConfig

__all__ = ["REDACTED", "FetchError", "Fetcher", "RedactSecrets", "hide_in_logs"]

log = logging.getLogger(__name__)

REDACTED = "<redacted>"

#: A secret shorter than this is not searched for: it would match inside
#: ordinary words and turn a log line into noise. Real keys are far longer.
_MIN_SECRET = 6


class RedactSecrets(logging.Filter):
    """Replaces every secret it has been given, in a log line, with `<redacted>`.

    For a key that travels in a URL: a source whose service wants the key in
    the path registers it (`hide_in_logs`), and the lines the shared client and
    httpx write about that URL no longer carry it. It never drops a line, and
    it changes one only when a registered secret is in it, as written or
    percent-encoded."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def add(self, secret: str) -> None:
        secret = secret.strip()
        if len(secret) >= _MIN_SECRET:
            self._secrets.update({secret, quote(secret, safe="")})

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        text = record.getMessage()
        hidden = text
        for secret in sorted(self._secrets, key=len, reverse=True):
            hidden = hidden.replace(secret, REDACTED)
        if hidden != text:
            record.msg, record.args = hidden, ()
        return True


#: The one filter, on the two loggers that write a request's URL: this
#: module's (the retry line) and httpx's own (one line per request, at INFO).
REDACTOR = RedactSecrets()
log.addFilter(REDACTOR)
logging.getLogger("httpx").addFilter(REDACTOR)


def hide_in_logs(secret: str) -> None:
    """Keep `secret` out of the log lines that show a request's URL."""
    REDACTOR.add(secret)


RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_BACKOFF = 30.0


class FetchError(RuntimeError):
    """A request failed after exhausting retries."""

    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"{url}: {detail}")
        self.url = url
        self.detail = detail


class Fetcher:
    """Async HTTP with bounded concurrency and sane retries.

    Used as an async context manager so the connection pool is always closed,
    including on the error paths.
    """

    def __init__(self, cfg: HTTPConfig | None = None) -> None:
        self.cfg = cfg or HTTPConfig()
        self._sem = asyncio.Semaphore(self.cfg.max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.timeout),
            headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json"},
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=self.cfg.max_concurrent * 2,
                max_keepalive_connections=self.cfg.max_concurrent,
            ),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "Fetcher must be used as an async context manager"
            raise RuntimeError(msg)
        return self._client

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return min(float(raw), _MAX_BACKOFF)
        except ValueError:
            return None

    def _backoff(self, attempt: int) -> float:
        """Exponential with full jitter, which avoids the retry thundering herd
        you get when twenty sources all fail against the same flaky host."""
        ceiling = min(_MAX_BACKOFF, 2.0**attempt)
        return random.uniform(0, ceiling)

    async def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """One request with bounded concurrency and retries, or FetchError.

        Shared by fetch_json and fetch_text so the retry policy exists once.
        """
        last = "unknown error"
        async with self._sem:
            for attempt in range(self.cfg.max_retries + 1):
                try:
                    r = await self.client.request(
                        method, url, params=params, json=json_body, headers=headers
                    )
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last = f"{type(e).__name__}: {e}"
                else:
                    if r.status_code < 400:
                        return r
                    last = f"HTTP {r.status_code}"
                    if r.status_code not in RETRY_STATUS:
                        # 404 on a wrong slug should fail immediately and loudly.
                        raise FetchError(url, last)
                    wait = self._retry_after(r) or self._backoff(attempt)
                    if attempt < self.cfg.max_retries:
                        log.debug("retrying %s in %.1fs (%s)", url, wait, last)
                        await asyncio.sleep(wait)
                    continue

                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

        raise FetchError(url, f"{last} after {self.cfg.max_retries + 1} attempts")

    async def fetch_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        r = await self._request(
            url, params=params, method=method, json_body=json_body, headers=headers
        )
        try:
            return r.json()
        except ValueError as e:
            raise FetchError(url, f"bad JSON: {e}") from e

    async def fetch_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Raw body, for the sources that parse HTML or XML rather than JSON."""
        r = await self._request(url, params=params, headers=headers)
        return r.text
