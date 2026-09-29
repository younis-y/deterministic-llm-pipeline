"""rolescan: a typed, async, plugin-based personal job scanner."""

from __future__ import annotations

from rolescan.config import Config
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.pipeline import ScanResult, run_scan

__version__ = "2.2.0"

__all__ = [
    "Config",
    "FitVerdict",
    "Job",
    "ScanResult",
    "ScoredJob",
    "Verdict",
    "__version__",
    "run_scan",
]
