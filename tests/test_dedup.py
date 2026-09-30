from __future__ import annotations

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
