from __future__ import annotations

import pytest
from pydantic import ValidationError

from rolescan.config import LLMConfig, ProfileConfig, RulesConfig
from rolescan.models import JobField, Level


def test_rules_are_optional_and_default_off() -> None:
    assert ProfileConfig().rules is None


def test_rules_parse_from_yaml_shaped_dict() -> None:
    p = ProfileConfig.model_validate({"rules": {
        "max_years_required": 2,
        "allowed_levels": ["graduate_entry", "junior", "not_stated"],
        "student_only": "skip",
        "allowed_fields": ["data_engineering", "ai_llm", "data_science"],
    }})
    assert p.rules is not None
    assert p.rules.max_years_required == 2
    assert Level.not_stated in p.rules.allowed_levels
    assert p.rules.allowed_fields == [JobField.data_engineering, JobField.ai_llm, JobField.data_science]


def test_unknown_rule_key_or_value_is_refused() -> None:
    with pytest.raises(ValidationError):
        RulesConfig.model_validate({"max_years": 2})
    with pytest.raises(ValidationError):
        RulesConfig.model_validate({"student_only": "maybe"})


def test_llm_mode_defaults_to_facts_and_accepts_judge() -> None:
    assert LLMConfig().mode == "facts"
    assert LLMConfig(mode="judge").mode == "judge"
    assert LLMConfig().enricher == ""


def test_2_5_0_rule_keys_parse_and_default_off() -> None:
    assert RulesConfig().max_graduation_year is None
    assert RulesConfig().level_from_title_only is False
    r = RulesConfig.model_validate({"max_graduation_year": 2027, "level_from_title_only": True})
    assert r.max_graduation_year == 2027 and r.level_from_title_only is True
