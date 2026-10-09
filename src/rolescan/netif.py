"""The IPv4 address of a named network interface, read when it is needed.

Used by `email.bind_interface` (2.5.10). On a machine where a VPN tunnel blocks
the mail ports while HTTPS passes, binding the SMTP socket to the address of the
interface that reaches the internet directly routes the mail around the tunnel.

The address is looked up at every send and never cached: a laptop that changes
network gets a new one, and a cached address would bind to one that is gone.
Standard library only: the kernel's SIOCGIFADDR request on Linux, the system
`ifconfig` on macOS and the BSDs.
"""

from __future__ import annotations

import ipaddress
import re
import shutil
import socket
import struct
import subprocess
import sys

__all__ = ["interface_ipv4", "parse_ifconfig_ipv4"]

#: Captured once so a test can stand in for another platform.
_PLATFORM = sys.platform

#: Linux's SIOCGIFADDR: "get the interface's protocol address".
_SIOCGIFADDR = 0x8915

#: Letters, digits and the few separators real names use (`en0`, `eth0.100`,
#: `br-lan`, `eth0:1`), at most 15 characters (IFNAMSIZ is 16 with the NUL), and
#: never a leading `-`, so a name cannot be taken for an option by `ifconfig`.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,14}")

#: `inet 192.0.2.17 netmask ...` (BSD, macOS) or `inet addr:192.0.2.17 ...`
#: (net-tools). `inet6` has no space after `inet`, so it never matches.
_INET = re.compile(r"^\s*inet\s+(?:addr:)?(\S+)", re.MULTILINE)


def interface_ipv4(name: str) -> str:
    """The interface's current IPv4 address, as dotted text.

    Raises ValueError, naming the interface, when `name` cannot be an interface
    name or the interface has no IPv4 address: it is down, it has not been given
    a lease yet, or it has been renamed (a USB adapter or a Wi-Fi card can come
    back as another number).
    """
    if not _NAME.fullmatch(name):
        message = f"{name!r} is not a valid interface name"
        raise ValueError(message)
    address: str | None
    try:
        if _PLATFORM.startswith("linux"):
            address = _ioctl_ipv4(name)
        else:
            address = parse_ifconfig_ipv4(_ifconfig(name))
    except (OSError, subprocess.SubprocessError):
        # No such device, no address assigned, no ifconfig binary, or one that
        # exits non-zero for an interface it does not know.
        address = None
    if address is None:
        message = (
            f"interface {name!r} has no IPv4 address (is it up, and is that "
            "still its name?)"
        )
        raise ValueError(message)
    return address


def parse_ifconfig_ipv4(text: str) -> str | None:
    """The first IPv4 address in `ifconfig <name>` output, or None."""
    for match in _INET.finditer(text):
        try:
            return str(ipaddress.IPv4Address(match.group(1)))
        except ValueError:
            continue
    return None


def _ioctl_ipv4(name: str) -> str:
    import fcntl  # POSIX only; imported here so the module loads anywhere

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        request = struct.pack("256s", name.encode("ascii"))
        reply = fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, request)
    # struct ifreq: 16 bytes of name, then a sockaddr_in whose address starts
    # at offset 4 within the sockaddr.
    return socket.inet_ntoa(reply[20:24])


def _ifconfig(name: str) -> str:
    binary = shutil.which("ifconfig") or "/sbin/ifconfig"
    done = subprocess.run(  # fixed binary; the name was checked by the caller
        [binary, name],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return done.stdout
