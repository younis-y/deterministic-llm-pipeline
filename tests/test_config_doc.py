"""docs/config.md is generated from the config models and cannot go stale."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import BaseModel

from rolescan.config import (
    EmailConfig,
    HTTPConfig,
    LLMConfig,
    OutputConfig,
    ProfileConfig,
    RetentionConfig,
    RulesConfig,
    SourceEntry,
)
from rolescan.models import JobField, Level

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "config.md"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_config_doc", ROOT / "scripts" / "gen_config_doc.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_reference_is_current() -> None:
    assert DOC.read_text(encoding="utf-8") == _generator().render(), (
        "docs/config.md is stale: run `python scripts/gen_config_doc.py`"
    )


MODELS: dict[str, type[BaseModel]] = {
    "profile": ProfileConfig,
    "profile.rules": RulesConfig,
    "llm": LLMConfig,
    "http": HTTPConfig,
    "output": OutputConfig,
    "output.retention_days": RetentionConfig,
    "output.email": EmailConfig,
}


@pytest.mark.parametrize("prefix", MODELS)
def test_every_config_key_is_in_the_reference(prefix: str) -> None:
    doc = DOC.read_text(encoding="utf-8")
    for name in MODELS[prefix].model_fields:
        assert f"`{prefix}.{name}`" in doc, f"{prefix}.{name} is not documented"


def test_every_source_entry_key_is_in_the_reference() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for name in SourceEntry.model_fields:
        assert f"`sources[].{name}`" in doc


def test_the_keys_once_missing_from_the_docs_are_there() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for key in (
        "profile.excluded_locations",
        "profile.agencies",
        "profile.agency_penalty",
        "profile.max_age_days",
        "llm.api_key",
        "llm.max_tokens",
        "llm.cascade",
        "llm.temperature",
        "http.user_agent",
        "output.email.password",
        "sources[].enabled",
        "sources[].verified",
    ):
        assert f"`{key}`" in doc, key


def test_every_level_and_field_value_is_listed() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for member in (*Level, *JobField):
        assert f"`{member.value}`" in doc
    assert "`finance`" in doc


def test_the_reference_says_keywords_are_substring_matches() -> None:
    doc = DOC.read_text(encoding="utf-8")
    row = next(
        line for line in doc.splitlines() if line.startswith("| `profile.keywords`")
    )
    assert "SUBSTRING" in row
    assert "not a whole word" in row


def test_every_source_option_the_code_reads_is_in_the_reference() -> None:
    doc = DOC.read_text(encoding="utf-8")
    read = set()
    for path in sorted((ROOT / "src" / "rolescan" / "sources").glob("*.py")):
        read |= set(re.findall(r'options\.get\(\s*"(\w+)"', path.read_text()))
    assert read, "found no source options: the pattern is stale"
    for option in sorted(read):
        assert f"`{option}`" in doc, f"source option {option!r} is not documented"


def test_the_generator_refuses_an_undescribed_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _generator()
    trimmed = dict(module.DESCRIPTIONS)
    del trimmed["profile.agencies"]
    monkeypatch.setattr(module, "DESCRIPTIONS", trimmed)
    with pytest.raises(module.MissingDescription, match=re.escape("profile.agencies")):
        module.render()
