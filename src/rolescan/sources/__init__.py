"""Source plugins.

Importing this package registers every built-in source and then loads any
third-party ones advertised under the `rolescan.sources` entry-point group.
"""

from __future__ import annotations

from rolescan.sources import (  # noqa: F401 - import registers
    adzuna,
    aggregators,
    ats,
    structured,
    workday,
)
from rolescan.sources.base import (
    Source,
    available,
    get_source,
    load_plugins,
    register,
    strip_html,
)

load_plugins()

__all__ = [
    "Source",
    "available",
    "get_source",
    "load_plugins",
    "register",
    "strip_html",
]
