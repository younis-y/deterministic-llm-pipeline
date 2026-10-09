"""Every registered source says what its dates mean.

`dates_are_freshness` decides whether the age cutoff may drop a posting. It
defaults to True, which is right for an aggregator and wrong for an employer's
own board, where a requisition opened years ago and still listed is still
open. A new employer-board source that forgot to say so would be aged out
silently, so this test names every source and the value it must declare: a new
source fails here until its author states one."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rolescan.models import Job
from rolescan.pipeline import _drop_stale
from rolescan.sources import available

#: kind -> `dates_are_freshness`. False: an employer's own board, where the
#: listing's presence is the freshness signal. True: an aggregator, whose
#: `posted` is when the advert went up.
EXPECTED: dict[str, bool] = {
    "adzuna": True,
    "ashby": False,
    "greenhouse": False,
    "jooble": True,
    "lever": False,
    "reed": True,
    "smartrecruiters": False,
    "structured": False,
    "workable": False,
    "workable_search": True,
    "workday": False,
}


def _core_sources() -> dict[str, bool]:
    """The sources this package ships, not whatever plugin the environment
    has installed."""
    return {
        kind: cls.dates_are_freshness
        for kind, cls in available().items()
        if cls.__module__.startswith("rolescan.sources")
    }


def test_every_core_source_is_listed_with_its_expected_value() -> None:
    assert _core_sources() == EXPECTED


@pytest.mark.parametrize("kind", ["structured", "workday"])
def test_a_long_open_requisition_on_an_employer_board_is_not_aged_out(
    kind: str,
) -> None:
    """The two sources that inherited the aggregator default, so a 90-day
    cutoff threw away every posting dated before it."""
    opened = (datetime.now(UTC) - timedelta(days=2000)).date()
    job = Job(
        source=kind,
        company="Acme",
        title="Energy Data Scientist",
        location="London",
        url=f"https://careers.example.test/{kind}/1",
        description="Python.",
        posted=opened,
    )
    kept, dropped = _drop_stale([job], 90)
    assert dropped == 0
    assert kept == [job]
