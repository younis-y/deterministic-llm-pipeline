"""Binding the mail socket to a named interface (2.5.10).

A VPN tunnel can block every mail port (465, 587, 993) while HTTPS passes. The
send then waits out its timeout, the digest stays on disk and the scheduled run
exits 1. Binding the SMTP socket to the address of the interface that reaches
the internet directly routes around the tunnel.

Nothing here opens a socket: `smtplib.SMTP` and `smtplib.SMTP_SSL` are replaced
by a class that records how it was called.
"""

from __future__ import annotations

import smtplib
import ssl
import subprocess
import sys
from typing import Any, ClassVar

import pytest

from rolescan import netif
from rolescan.config import Config, EmailConfig
from rolescan.digest import EmailError, send_email
from rolescan.netif import interface_ipv4, parse_ifconfig_ipv4

# RFC 5737 documentation addresses: nothing here is a real host.
IP_A = "192.0.2.17"
IP_B = "198.51.100.9"

MACOS_IFCONFIG = """\
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\toptions=6460<TSO4,TSO6,CHANNEL_IO,PARTIAL_CSUM,ZEROINVERT_CSUM>
\tether 02:00:5e:10:00:01
\tinet6 fe80::1c2a:3b4c:5d6e:7f80%en0 prefixlen 64 secured scopeid 0xb
\tinet 192.0.2.17 netmask 0xffffff00 broadcast 192.0.2.255
\tnd6 options=201<PERFORMNUD,DAD>
\tmedia: autoselect
\tstatus: active
"""

MACOS_IFCONFIG_NO_IPV4 = """\
en0: flags=8822<BROADCAST,SMART,SIMPLEX,MULTICAST> mtu 1500
\tether 02:00:5e:10:00:01
\tinet6 fe80::1c2a:3b4c:5d6e:7f80%en0 prefixlen 64 secured scopeid 0xb
\tmedia: autoselect
\tstatus: inactive
"""

NET_TOOLS_IFCONFIG = """\
eth0      Link encap:Ethernet  HWaddr 02:00:5e:10:00:01
          inet addr:198.51.100.9  Bcast:198.51.100.255  Mask:255.255.255.0
          inet6 addr: fe80::1/64 Scope:Link
"""


class _FakeSMTP:
    """Stands in for `smtplib.SMTP`. Records how it was built and used."""

    log: ClassVar[list[dict[str, Any]]] = []
    implicit_tls = False
    fail_on: ClassVar[str] = ""
    failure: ClassVar[BaseException] = TimeoutError("timed out")

    def __init__(
        self,
        host: str,
        port: int = 0,
        local_hostname: str | None = None,
        *,
        timeout: float = 0,
        source_address: tuple[str, int] | None = None,
        context: ssl.SSLContext | None = None,
    ) -> None:
        self.record: dict[str, Any] = {
            "host": host,
            "port": port,
            "timeout": timeout,
            "source_address": source_address,
            "implicit_tls": self.implicit_tls,
            "context": context,
            "starttls": False,
            "starttls_context": None,
            "login": None,
            "sent": 0,
            "closed": False,
            "quit": False,
        }
        type(self).log.append(self.record)
        self._maybe_fail("connect")

    def _maybe_fail(self, step: str) -> None:
        if type(self).fail_on == step:
            raise type(self).failure

    def __enter__(self) -> _FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        self.record["quit"] = True
        self.close()

    def close(self) -> None:
        self.record["closed"] = True

    def starttls(self, *, context: ssl.SSLContext | None = None) -> None:
        assert not self.implicit_tls, "STARTTLS on a connection that is already TLS"
        self.record["starttls"] = True
        self.record["starttls_context"] = context
        self._maybe_fail("starttls")

    def login(self, user: str, password: str) -> None:
        self.record["login"] = user
        self._maybe_fail("login")

    def send_message(self, msg: object) -> None:
        self.record["sent"] += 1


class _FakeSMTPSSL(_FakeSMTP):
    implicit_tls = True


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSMTP]:
    _FakeSMTP.log = []
    _FakeSMTP.fail_on = ""
    _FakeSMTP.failure = TimeoutError("timed out")
    monkeypatch.setattr("rolescan.digest.smtplib.SMTP", _FakeSMTP)
    monkeypatch.setattr("rolescan.digest.smtplib.SMTP_SSL", _FakeSMTPSSL)
    return _FakeSMTP


def _cfg(**over: Any) -> EmailConfig:
    base: dict[str, Any] = {
        "enabled": True,
        "smtp_host": "smtp.example.test",
        "username": "me@example.test",
        "password": "not-a-real-password",
        "to": "me@example.test",
    }
    return EmailConfig(**{**base, **over})


# --- config ------------------------------------------------------------------


def test_the_defaults_bind_nothing_and_keep_starttls_on_587() -> None:
    cfg = EmailConfig()
    assert cfg.bind_interface == ""
    assert cfg.smtp_port == 587


def test_bind_interface_is_read_from_the_config_file() -> None:
    cfg = Config.model_validate({"output": {"email": {"bind_interface": "en0"}}})
    assert cfg.output.email.bind_interface == "en0"


# --- sending: unchanged behaviour --------------------------------------------


def test_without_bind_interface_nothing_is_bound_and_starttls_is_used(
    smtp: type[_FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str) -> str:
        raise AssertionError("no interface is configured, so none is looked up")

    monkeypatch.setattr("rolescan.digest.interface_ipv4", refuse)
    assert send_email("body", _cfg()) is True
    (call,) = smtp.log
    assert call["host"] == "smtp.example.test"
    assert call["port"] == 587
    assert call["source_address"] is None
    assert call["implicit_tls"] is False
    assert call["starttls"] is True
    assert call["login"] == "me@example.test"
    assert call["sent"] == 1


def test_a_disabled_emailer_connects_to_nothing(smtp: type[_FakeSMTP]) -> None:
    assert send_email("body", _cfg(enabled=False)) is False
    assert smtp.log == []


# --- sending: the bound socket -----------------------------------------------


def test_bind_interface_binds_the_socket_to_that_interfaces_address(
    smtp: type[_FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def lookup(name: str) -> str:
        asked.append(name)
        return IP_A

    monkeypatch.setattr("rolescan.digest.interface_ipv4", lookup)
    assert send_email("body", _cfg(bind_interface="en0")) is True
    (call,) = smtp.log
    assert asked == ["en0"]
    assert call["source_address"] == (IP_A, 0)
    assert call["port"] == 587
    assert call["implicit_tls"] is False
    assert call["starttls"] is True
    assert call["sent"] == 1


def test_the_address_is_looked_up_at_every_send(
    smtp: type[_FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A laptop that changes network gets a new address; a value cached at
    config load would bind to one that no longer exists."""
    addresses = iter([IP_A, IP_B])
    monkeypatch.setattr("rolescan.digest.interface_ipv4", lambda name: next(addresses))
    cfg = _cfg(bind_interface="en0")
    send_email("one", cfg)
    send_email("two", cfg)
    assert [c["source_address"] for c in smtp.log] == [(IP_A, 0), (IP_B, 0)]


def test_port_465_uses_implicit_tls_not_starttls(smtp: type[_FakeSMTP]) -> None:
    assert send_email("body", _cfg(smtp_port=465)) is True
    (call,) = smtp.log
    assert call["implicit_tls"] is True
    assert call["starttls"] is False
    assert call["port"] == 465
    assert call["source_address"] is None
    assert call["login"] == "me@example.test"
    assert call["sent"] == 1


def test_port_465_verifies_the_servers_certificate(smtp: type[_FakeSMTP]) -> None:
    """smtplib's own default context for SMTP_SSL does not check the
    certificate; a new code path should not start out unverified."""
    send_email("body", _cfg(smtp_port=465))
    context = smtp.log[0]["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_starttls_verifies_the_servers_certificate(smtp: type[_FakeSMTP]) -> None:
    """`starttls()` with no context does not check the certificate or the
    host name, so a machine in the path could read the login and the digest.
    The context is the one `ssl.create_default_context()` makes."""
    send_email("body", _cfg(smtp_port=587))
    context = smtp.log[0]["starttls_context"]
    assert isinstance(context, ssl.SSLContext), "starttls was called with no context"
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_every_non_implicit_port_verifies_on_starttls(smtp: type[_FakeSMTP]) -> None:
    for port in (25, 587, 2525):
        smtp.log.clear()
        send_email("body", _cfg(smtp_port=port))
        context = smtp.log[0]["starttls_context"]
        assert isinstance(context, ssl.SSLContext), port
        assert context.verify_mode == ssl.CERT_REQUIRED, port


def test_a_certificate_failure_at_starttls_reaches_the_caller_unchanged(
    smtp: type[_FakeSMTP],
) -> None:
    """A self-signed relay now fails here. The error is the certificate's own,
    not a port blamed on the network, so the fix (a trusted certificate) is
    what the reader is pointed at."""
    smtp.fail_on = "starttls"
    smtp.failure = ssl.SSLCertVerificationError("self-signed certificate")
    with pytest.raises(ssl.SSLCertVerificationError, match="self-signed"):
        send_email("body", _cfg(smtp_port=587))


def test_port_465_binds_the_same_way(
    smtp: type[_FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rolescan.digest.interface_ipv4", lambda name: IP_A)
    send_email("body", _cfg(smtp_port=465, bind_interface="en0"))
    (call,) = smtp.log
    assert call["implicit_tls"] is True
    assert call["source_address"] == (IP_A, 0)


def test_other_ports_keep_starttls(smtp: type[_FakeSMTP]) -> None:
    send_email("body", _cfg(smtp_port=2525))
    (call,) = smtp.log
    assert call["implicit_tls"] is False
    assert call["starttls"] is True


# --- sending: an interface with no IPv4 --------------------------------------


def test_an_interface_without_an_ipv4_is_an_error_that_names_it(
    smtp: type[_FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    def lookup(name: str) -> str:
        raise ValueError(f"interface {name!r} has no IPv4 address")

    monkeypatch.setattr("rolescan.digest.interface_ipv4", lookup)
    with pytest.raises(EmailError, match=r"email\.bind_interface.*'en7'"):
        send_email("body", _cfg(bind_interface="en7"))
    assert smtp.log == [], "no connection is attempted from an unknown address"


# --- sending: the mail port does not answer ----------------------------------


@pytest.mark.parametrize("step", ["connect", "starttls"])
@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timed out"),
        OSError("[Errno 65] No route to host"),
        smtplib.SMTPServerDisconnected("Connection unexpectedly closed: timed out"),
    ],
    ids=["timeout", "oserror", "banner-never-arrives"],
)
def test_a_port_that_does_not_answer_says_what_to_try(
    smtp: type[_FakeSMTP], step: str, failure: Exception
) -> None:
    smtp.fail_on = step
    smtp.failure = failure
    with pytest.raises(EmailError) as err:
        send_email("body", _cfg())
    text = str(err.value)
    assert "mail port 587 on smtp.example.test did not answer" in text
    assert "a VPN or firewall may block mail ports" in text
    assert (
        "set email.bind_interface to the interface that reaches the internet "
        "directly (for example en0)"
    ) in text
    assert str(failure) in text, "the underlying cause is kept"
    assert err.value.__cause__ is failure
    if step == "starttls":
        assert smtp.log[0]["closed"] is True, "the half-open connection is closed"


def test_a_port_465_that_does_not_answer_names_port_465(
    smtp: type[_FakeSMTP],
) -> None:
    smtp.fail_on = "connect"
    with pytest.raises(
        EmailError, match=r"mail port 465 on smtp\.example\.test did not"
    ):
        send_email("body", _cfg(smtp_port=465))


def test_a_bound_send_that_does_not_answer_says_which_interface(
    smtp: type[_FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rolescan.digest.interface_ipv4", lambda name: IP_A)
    smtp.fail_on = "connect"
    with pytest.raises(EmailError, match="did not answer from interface en0"):
        send_email("body", _cfg(bind_interface="en0"))


def test_a_refused_login_is_not_blamed_on_the_network(smtp: type[_FakeSMTP]) -> None:
    """`SMTPException` subclasses `OSError`, so a catch for network trouble
    would otherwise relabel a wrong password as a blocked port."""
    smtp.fail_on = "login"
    smtp.failure = smtplib.SMTPAuthenticationError(535, b"bad credentials")
    with pytest.raises(smtplib.SMTPAuthenticationError):
        send_email("body", _cfg())


def test_a_certificate_that_does_not_verify_is_not_blamed_on_the_network(
    smtp: type[_FakeSMTP],
) -> None:
    """The server answered; the handshake failed because of what it said."""
    smtp.fail_on = "connect"
    smtp.failure = ssl.SSLCertVerificationError(1, "certificate verify failed")
    with pytest.raises(ssl.SSLCertVerificationError):
        send_email("body", _cfg(smtp_port=465))


def test_a_server_that_does_not_offer_starttls_is_not_blamed_on_the_network(
    smtp: type[_FakeSMTP],
) -> None:
    smtp.fail_on = "starttls"
    smtp.failure = smtplib.SMTPNotSupportedError("STARTTLS extension not supported")
    with pytest.raises(smtplib.SMTPNotSupportedError):
        send_email("body", _cfg())
    assert smtp.log[0]["closed"] is True


# --- netif: reading an interface's IPv4 --------------------------------------


def test_the_macos_ifconfig_parser_takes_the_inet_line_not_inet6() -> None:
    assert parse_ifconfig_ipv4(MACOS_IFCONFIG) == IP_A


def test_the_net_tools_ifconfig_format_is_read_too() -> None:
    assert parse_ifconfig_ipv4(NET_TOOLS_IFCONFIG) == IP_B


def test_ifconfig_output_with_only_inet6_has_no_ipv4() -> None:
    assert parse_ifconfig_ipv4(MACOS_IFCONFIG_NO_IPV4) is None
    assert parse_ifconfig_ipv4("") is None


def test_an_inet_line_with_a_malformed_address_is_not_an_address() -> None:
    assert parse_ifconfig_ipv4("\tinet 999.1.1.1 netmask 0xff000000\n") is None


def test_the_bsd_path_runs_ifconfig_on_the_named_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_ifconfig(name: str) -> str:
        seen.append(name)
        return MACOS_IFCONFIG

    monkeypatch.setattr(netif, "_PLATFORM", "darwin")
    monkeypatch.setattr(netif, "_ifconfig", fake_ifconfig)
    assert interface_ipv4("en0") == IP_A
    assert seen == ["en0"]


def test_the_linux_path_reads_the_address_from_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(netif, "_PLATFORM", "linux")
    monkeypatch.setattr(netif, "_ioctl_ipv4", lambda name: IP_B)

    def never(name: str) -> str:
        raise AssertionError("ifconfig is not needed on Linux")

    monkeypatch.setattr(netif, "_ifconfig", never)
    assert interface_ipv4("eth0") == IP_B


def test_an_interface_that_is_down_is_an_error_that_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(netif, "_PLATFORM", "darwin")
    monkeypatch.setattr(netif, "_ifconfig", lambda name: MACOS_IFCONFIG_NO_IPV4)
    with pytest.raises(ValueError, match=r"'en0'.*no IPv4 address"):
        interface_ipv4("en0")


def test_an_interface_that_does_not_exist_is_an_error_that_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gone(name: str) -> str:
        raise subprocess.CalledProcessError(1, ["ifconfig", name])

    monkeypatch.setattr(netif, "_PLATFORM", "darwin")
    monkeypatch.setattr(netif, "_ifconfig", gone)
    with pytest.raises(ValueError, match=r"'en9'.*no IPv4 address"):
        interface_ipv4("en9")


def test_a_linux_interface_the_kernel_does_not_know_is_an_error_that_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_device(name: str) -> str:
        raise OSError(19, "No such device")

    monkeypatch.setattr(netif, "_PLATFORM", "linux")
    monkeypatch.setattr(netif, "_ioctl_ipv4", no_device)
    with pytest.raises(ValueError, match=r"'wlan9'.*no IPv4 address"):
        interface_ipv4("wlan9")


@pytest.mark.parametrize(
    "name", ["", "-a", "en0; reboot", "en 0", "a-name-over-fifteen", "../en0"]
)
def test_a_name_that_cannot_be_an_interface_is_refused_before_anything_runs(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(n: str) -> str:
        raise AssertionError("no lookup for a malformed name")

    monkeypatch.setattr(netif, "_ifconfig", never)
    monkeypatch.setattr(netif, "_ioctl_ipv4", never)
    with pytest.raises(ValueError, match="not a valid interface name"):
        interface_ipv4(name)


@pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")),
    reason="needs a POSIX loopback interface",
)
def test_the_real_lookup_finds_the_loopback_address() -> None:
    """The one test that does not stub the operating system: ioctl on Linux,
    the `ifconfig` binary on macOS."""
    loopback = "lo" if sys.platform.startswith("linux") else "lo0"
    assert interface_ipv4(loopback) == "127.0.0.1"
