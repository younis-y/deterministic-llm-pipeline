"""The suite must never reach the network. Found 2026-10-07: one test passed
only because api.anthropic.com rejected its fake key with a 401, so every full
run sent a real request. The autouse blocker in conftest makes that a loud
failure instead."""

from __future__ import annotations

import socket

import pytest


def test_remote_dns_is_refused() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("api.anthropic.com", 443)


def test_localhost_still_resolves() -> None:
    assert socket.getaddrinfo("localhost", 11434)
    assert socket.getaddrinfo("127.0.0.1", 11434)
