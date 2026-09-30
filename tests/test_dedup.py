from __future__ import annotations

import pytest

from rolescan.dedup import city_key, company_key, merge_near_duplicates
from rolescan.models import Job


def _job(company: str, title: str, location: str, url: str, desc: str = "d") -> Job:
    return Job(
        source="adzuna",
        company=company,
        title=title,
        location=location,
        url=url,
        description=desc,
    )


def test_the_real_sanderson_pair_merges() -> None:
    # Both appeared in the 2026-09-30 digest as separate roles.
    a = _job(
        "Sanderson Recruitment",
        "Data Engineer",
        "South East London, London",
        "https://www.adzuna.co.uk/jobs/land/ad/5903086047",
        "short",
    )
    b = _job(
        "Sanderson Recruitment",
        "Data Engineer",
        "London, UK",
        "https://www.adzuna.co.uk/jobs/details/5903493155",
        "a longer description",
    )
    out = merge_near_duplicates([a, b])
    assert len(out) == 1
    assert out[0].description == "a longer description"


def test_same_title_at_two_companies_stays_separate() -> None:
    a = _job("Drax", "Data Engineer", "London", "https://x/1")
    b = _job("Octopus Energy", "Data Engineer", "London", "https://x/2")
    assert len(merge_near_duplicates([a, b])) == 2


def test_same_company_in_two_cities_stays_separate() -> None:
    a = _job("Vitol", "Data Engineer", "London, UK", "https://x/1")
    b = _job("Vitol", "Data Engineer", "Dubai, United Arab Emirates", "https://x/2")
    assert len(merge_near_duplicates([a, b])) == 2


def test_senior_and_plain_title_stay_separate() -> None:
    a = _job("Drax", "Data Engineer", "London", "https://x/1")
    b = _job("Drax", "Senior Data Engineer", "London", "https://x/2")
    assert len(merge_near_duplicates([a, b])) == 2


def test_empty_company_never_merges() -> None:
    a = _job("", "Data Engineer", "London", "https://x/1")
    b = _job("", "Data Engineer", "London", "https://x/2")
    assert len(merge_near_duplicates([a, b])) == 2


def test_arabic_names_survive_and_merge() -> None:
    a = _job(
        "شركة أرامكو",
        "مهندس بيانات",
        "الرياض",
        "https://x/1",
        "short",
    )
    b = _job(
        "شركة أرامكو",
        "مهندس بيانات",
        "الرياض",
        "https://x/2",
        "longer text",
    )
    c = _job(
        "شركة أرامكو", "محلل مالي", "الرياض", "https://x/3"
    )
    assert company_key("شركة أرامكو") == "شركة أرامكو"
    out = merge_near_duplicates([a, b, c])
    assert len(out) == 2
    assert {j.url for j in out} == {"https://x/2", "https://x/3"}


def test_result_keeps_input_order_and_is_order_independent() -> None:
    a = _job("Drax", "Data Engineer", "London", "https://x/1", "short")
    b = _job("Drax", "Data Engineer", "London, UK", "https://x/2", "much longer")
    c = _job("Ovo", "Analyst", "Bristol", "https://x/3")
    assert [j.url for j in merge_near_duplicates([c, a, b])] == [
        "https://x/3",
        "https://x/2",
    ]
    assert [j.url for j in merge_near_duplicates([b, c, a])] == [
        "https://x/2",
        "https://x/3",
    ]


def test_keys() -> None:
    assert company_key("Areti Group | B Corp™") == "areti group"
    assert company_key("Acme Ltd.") == "acme"
    assert city_key("South East London, London") == "london"
    assert city_key("London Area, United Kingdom") == "london"
    assert city_key("United Kingdom") == ""
    assert city_key("") == ""


def _pair_merges(t1: str, t2: str) -> bool:
    a = _job("Drax", t1, "London", "https://x/1", "longer description")
    b = _job("Drax", t2, "London", "https://x/2", "short")
    return len(merge_near_duplicates([a, b])) == 1


@pytest.mark.parametrize(
    ("t1", "t2"),
    [
        ("Data Engineer I", "Data Engineer II"),
        ("6 month FTC", "12 month FTC"),
        ("Senior Data Engineer", "Senior Data Engineer II"),
        ("MLE", "MLE - NLP"),
    ],
)
def test_different_roles_with_similar_titles_stay_separate(t1: str, t2: str) -> None:
    # Each pair was merged by a similarity-only rule, and `seen` then hid the
    # loser permanently.
    assert not _pair_merges(t1, t2)
    assert not _pair_merges(t2, t1)


@pytest.mark.parametrize(
    ("t1", "t2"),
    [
        ("Data Engineer", "Data Engineers"),
        ("Data Engineer", "Data Engineer"),
        ("Senior Data Engineer", "senior data engineer"),
    ],
)
def test_same_role_written_slightly_differently_merges(t1: str, t2: str) -> None:
    assert _pair_merges(t1, t2)
    assert _pair_merges(t2, t1)
