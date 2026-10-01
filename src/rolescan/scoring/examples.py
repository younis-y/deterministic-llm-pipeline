"""Worked examples for the facts-mode system prompt (`llm.facts_examples_file`).

A model shown a handful of postings with the facts a careful reader would
extract from them follows the extraction rules more closely than one given the
rules alone. The examples are the caller's own data - they describe one
candidate's market and judgement - so the core ships none and carries no
vocabulary for them: it only loads, validates and renders a file the caller
points it at.

They go into the SYSTEM prompt, after the instructions, rather than into each
user message, so they sit inside the prefix the Anthropic backend caches and
are paid for once per cache window rather than once per posting.

File format, a YAML list:

    - title: str
      company: str
      description: str      # an excerpt is fine
      facts:                # a PostingFacts mapping, as the model returns it
        level: {value: ..., quote: ...}
        graduation_year: {value: ..., quote: ...}
        fit_score: ...
        reason: ...
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from rolescan.scoring.facts import PostingFacts

__all__ = ["FactsExample", "load_facts_examples", "render_facts_examples"]

#: Lead-in for the rendered block. Says outright that the example postings are
#: not the posting to extract from: the quote guard would discard a fact quoted
#: from an example anyway, but a model that never makes that mistake does not
#: lose the fact in the first place.
_HEADER = (
    "Worked examples. Each shows an example posting and the facts to extract "
    "from it, as JSON. They illustrate the rules above; they are not the "
    "posting you are given. Quote only from the posting in the user message."
)

#: Provenance `resolve_level` sets after the call. Never shown to the model as
#: part of what it returns.
_NOT_MODEL_OUTPUT: dict[str, set[str]] = {"level": {"source"}}


class FactsExample(BaseModel):
    """One worked example: a posting excerpt and the facts it should yield."""

    model_config = ConfigDict(extra="forbid")

    title: str
    company: str
    description: str
    facts: PostingFacts


def load_facts_examples(path: Path) -> list[FactsExample]:
    """Load and validate every example in `path`, or raise `ValueError`.

    Each example's `facts` is validated as a `PostingFacts`, so an example
    that could never have been a model answer (an unknown key, an
    out-of-range score, a value not in an enum) fails here, naming the file
    and the example by position and title, rather than teaching the model a
    shape the schema then refuses.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        msg = f"llm.facts_examples_file {path}: cannot be read: {e}"
        raise ValueError(msg) from e
    if not isinstance(data, list) or not data:
        msg = (
            f"llm.facts_examples_file {path}: expected a non-empty YAML list of "
            "examples (title, company, description, facts)"
        )
        raise ValueError(msg)
    examples: list[FactsExample] = []
    for i, raw in enumerate(data, start=1):
        title = raw.get("title", "") if isinstance(raw, dict) else ""
        try:
            examples.append(FactsExample.model_validate(raw))
        except ValidationError as e:
            msg = (
                f"llm.facts_examples_file {path}: example {i} ({title!r}) is "
                f"invalid: {e}"
            )
            raise ValueError(msg) from e
    return examples


def render_facts_examples(examples: list[FactsExample]) -> str:
    """The examples as a block for the end of the facts system prompt.

    Posting text first, then the expected JSON, which is exactly the shape the
    model returns (`PostingFacts`, every field present, provenance left out).
    No examples render as "", so the prompt is unchanged byte for byte.
    """
    if not examples:
        return ""
    parts = [_HEADER]
    for i, example in enumerate(examples, start=1):
        expected = example.facts.model_dump(mode="json", exclude=_NOT_MODEL_OUTPUT)
        parts.append(
            f'<example index="{i}">\n'
            "<example_posting>\n"
            f"Title: {example.title}\n"
            f"Company: {example.company}\n\n"
            f"{example.description.strip()}\n"
            "</example_posting>\n"
            "<expected_facts>\n"
            f"{json.dumps(expected, indent=2, ensure_ascii=False)}\n"
            "</expected_facts>\n"
            "</example>"
        )
    return "\n\n".join(parts)
