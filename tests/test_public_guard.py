"""Nothing personal is committed to this public repository.

Every file git tracks (or would track: untracked and not ignored) is read for
three kinds of leak:

- an e-mail address that is not on a reserved example domain;
- a home-directory path (`/Users/<name>`, `/home/<name>`);
- a phone number.

This file holds no list of names, hashed or otherwise: a hash of one word lets
anyone confirm a guess. To check your own names, employers and projects, put
them one per line (case-insensitive, matched as substrings, `#` starts a
comment) in a file outside this repository and name it:

    ROLESCAN_GUARD_DENYLIST=~/private/deny.txt pytest tests/test_public_guard.py

Unset, that part is skipped, which is how CI runs. Set to a file that cannot be
read, the test fails: a mistyped path must not switch the check off.

A fixture needing an address uses an example domain (`example.com`, `.org`,
`.net`, or any `.test`, `.example`, `.invalid` name), a path under `/tmp`, and an
invented person or company.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Names an optional file of extra terms to refuse (see the module docstring).
DENYLIST_ENV = "ROLESCAN_GUARD_DENYLIST"

_TOKEN = re.compile(r"[a-z0-9]+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_HOME_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")
_PHONE = re.compile(
    r"(?<![\w.])(?:"
    r"\+\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){2,4}"  # international, with a plus sign
    r"|\(?0\d{2,4}\)?[ .-]\d{3,4}[ .-]\d{3,4}"  # national, leading zero
    r"|\(?0\d{3,4}\)?[ .-]\d{5,7}"  # national mobile, two groups
    r"|0\d{9,10}"  # national, no separators: a leading 0 and 10-11 digits
    r"|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}"  # three-three-four digits
    r")(?![\w.])"
)
_EXAMPLE_DOMAINS = frozenset({"example.com", "example.org", "example.net"})
_RESERVED_SUFFIXES = (".test", ".example", ".invalid", ".localhost")

#: A file this large is data, not prose.
_MAX_BYTES = 2_000_000


def _allowed_domain(domain: str) -> bool:
    name = domain.lower()
    if name in _EXAMPLE_DOMAINS or name.endswith(
        tuple(f".{d}" for d in _EXAMPLE_DOMAINS)
    ):
        return True
    return name.endswith(_RESERVED_SUFFIXES)


def load_denylist() -> list[str]:
    """The lower-cased terms in the file `ROLESCAN_GUARD_DENYLIST` names, or
    [] when it is unset or empty. A file that is named and unreadable raises:
    a mistyped path must not turn the check off without a word."""
    name = os.environ.get(DENYLIST_ENV, "").strip()
    if not name:
        return []
    lines = Path(name).expanduser().read_text(encoding="utf-8").splitlines()
    terms = [line.strip().lower() for line in lines]
    return [t for t in terms if t and not t.startswith("#")]


def violations(text: str, denylist: Sequence[str] = ()) -> list[str]:
    """What in `text` a public file must not carry, as short descriptions.

    A denylist hit does not quote the term: the message reaches terminals and
    CI logs, and the list is private."""
    found = [
        f"e-mail address at {m.group(1)}"
        for m in _EMAIL.finditer(text)
        if not _allowed_domain(m.group(1))
    ]
    found += [f"home path {m.group(0)}" for m in _HOME_PATH.finditer(text)]
    found += [f"phone number {m.group(0)!r}" for m in _PHONE.finditer(text)]
    lowered = text.lower()
    found += [
        f"a term from {DENYLIST_ENV} (entry {i} of the list)"
        for i, term in enumerate(denylist, 1)
        if term and term.lower() in lowered
    ]
    return found


def _publishable_files() -> list[Path]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    names = [n for n in listing.decode().split("\0") if n]
    return [ROOT / n for n in names if (ROOT / n).is_file()]


def test_no_tracked_file_carries_personal_data() -> None:
    denylist = load_denylist()
    problems: list[str] = []
    for path in _publishable_files():
        if path.stat().st_size > _MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        problems += [
            f"{path.relative_to(ROOT)}: {p}" for p in violations(text, denylist)
        ]
    assert not problems, "personal data in a public file:\n" + "\n".join(problems)


# --- the guard itself ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "write to " + "someone@" + "corp-mail.io",
        "mail " + "a.b@" + "mail.example.co.uk",
        "see /Us" + "ers/someone/project",
        "see /ho" + "me/someone/.config",
        "call +44 " + "20 7946 0958",
        "call 020 " + "7946 0958",
        "call 415-" + "555-0100",
        "mobile 07700 " + "900123",
        "mobile 07700-" + "900123",
        "mobile (07700) " + "900123",
        "mobile 07700" + "900123",
        "landline 020" + "79460958",
        "call +44" + "7700900123",
        "call +4479" + "00900123",
    ],
)
def test_the_guard_catches_a_leak(text: str) -> None:
    assert violations(text), text


@pytest.mark.parametrize(
    "text",
    [
        "someone@" + "example.com",
        "someone@" + "mail.example.org",
        "someone@" + "inbox.test",
        "@pytest.mark.parametrize and @respx.mock",
        "see /tmp/scratch/digests",
        "num_ctx: 12288, 2026-10-07, version 2.5.10, 192.0.2.17",
        "sha 4b9e2848 and 0123456789ab",
        "id 123456789012 and build 1234567890",
        "epoch 1760000000 and stamp 20261007143000",
        "ratio 0.5 to 1 in 0.125-0.25, 2026-10-07T14:30",
        "a leading-zero code 000123456 is nine digits",
    ],
)
def test_the_guard_passes_clean_text(text: str) -> None:
    assert violations(text) == [], text


def test_the_public_test_holds_no_hashed_denylist() -> None:
    """A single-token SHA-256 lets anyone confirm a guessed name, so this file
    keeps no digest list. Extra terms live in a file the checker names in the
    environment, outside the repository."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{64}\b", source), "a SHA-256 digest"


def test_a_denylist_term_is_caught_case_insensitively() -> None:
    """Planted with a sentinel of our own, so this file names no one."""
    assert violations("mentions ZzSentinelZz, once", ["zzsentinelzz"])
    assert violations("a zzsentinelzz-x token", ["zzsentinelzz"]), "a substring"
    assert violations("two words: Yy First yy", ["yy first"])
    assert violations("mentions nobody", ["zzsentinelzz"]) == []
    assert violations("mentions zzsentinelzz") == [], "no list, no check"


def test_a_denylist_hit_does_not_echo_the_term() -> None:
    """The failure text lands in terminals and CI logs."""
    (hit,) = violations("mentions zzsentinelzz", ["zzsentinelzz"])
    assert "zzsentinelzz" not in hit.lower()


def test_the_denylist_is_read_from_the_file_the_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = tmp_path / "deny.txt"
    listing.write_text("ZzSentinelZz\n\n# a comment\n  Yy First  \n", encoding="utf-8")
    monkeypatch.setenv(DENYLIST_ENV, str(listing))
    assert load_denylist() == ["zzsentinelzz", "yy first"]


def test_an_unset_denylist_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DENYLIST_ENV, raising=False)
    assert load_denylist() == []
    monkeypatch.setenv(DENYLIST_ENV, "")
    assert load_denylist() == []


def test_a_denylist_that_cannot_be_read_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped path must not quietly switch the check off."""
    monkeypatch.setenv(DENYLIST_ENV, str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        load_denylist()


def test_the_adrs_name_no_owner() -> None:
    """A decision record says what was decided and why, not who by or on whose
    data."""
    for path in sorted((ROOT / "docs" / "adr").glob("*.md")):
        words = set(_TOKEN.findall(path.read_text(encoding="utf-8").lower()))
        assert not words & {"owner", "owners"}, path.name
