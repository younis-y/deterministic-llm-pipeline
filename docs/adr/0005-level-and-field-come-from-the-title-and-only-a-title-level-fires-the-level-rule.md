# ADR-0005: Level and field are read from the title; with `level_from_title_only`, only a title level can skip

**Date**: 2026-10-01
**Status**: accepted
**Deciders**: maintainers

## Context

The local model answered `not_stated` while quoting "Senior Data & BI Engineer",
and skipped a two-years role on the one-word quote "Senior" lifted from a
sentence about stakeholders. Worse, under a title-only rule, a model that
inferred `mid` for the title "Data Engineer" and quoted the title as its evidence
caused a skip. A verbatim quote of the title is not the same as the title naming
a level. A skipped role is hidden permanently, so a skip must rest on the
advert's own words.

## Decision

1. The title is the better evidence. `resolve_level` sets the level from a level
   word in the title (`_LEVEL_WORDS`) and records `source="title"`. Otherwise the
   model's level survives as `source="text"` only with a verified quote of two or
   more words. Otherwise it is `not_stated`. The source is overwritten every
   time, so a model can never claim `title` for itself.
2. `rules.level_from_title_only: true` means only `source == "title"` can fire the
   level rule. A level the model states is ignored, even when its quote is copied
   from the title.
3. `resolve_field` does the same for field, from `_FIELD_WORDS` (the first match
   wins), and never returns `other` from a title: a missing word is not evidence
   for a field.
4. Level-word precedence: graduate or entry-level beats junior beats everything
   else; lead beats senior; senior beats mid. "manager" is not a level word.
   `staff` and `lead` count only next to a role word. `mid-senior` is senior, and
   a bare `mid` is not a level word.

## Alternatives Considered

### Alternative 1: Trust the model's level when it has a quote
- **Pros**: no word tables.
- **Cons**: it shipped, and failed in both directions (above).
- **Why not**: measured defects.

### Alternative 2: Derive the level from years of experience too
- **Pros**: catches seniority the title does not name.
- **Cons**: years have their own rule (`max_years_required`), and the two are
  meant to be judged independently.
- **Why not**: the rules stay separate (ADR-0002).

## Consequences

### Positive
- A skip for level always quotes the title, so it is checkable at a glance.
- On a labelled evaluation the seniority rule was right on nearly every case it
  fired on.

### Negative
- The word tables are a maintenance surface. A change to one is a change to what
  a fact means, and `tests/test_facts_key_pin.py` forces a conscious
  `FACTS_KEY_VERSION` decision when a table moves (ADR-0008).
- A senior role whose title names no level ("Data Engineer") and whose advert
  says "8 years" is not skipped by the level rule. Only the years rule can catch
  it, and only if the years resolver reads the sentence.

### Risks
- When `allowed_levels` excludes `mid`, a new mid-level title word silently starts
  hiding roles. "Associate" is deliberately not a level word.
- The title tables encode English only. A title in another language names no level
  and no field.
- Banking grades are not level words, and one is read backwards. "Assistant Vice
  President, Data Engineering" resolves to `junior`, because the `assistant` word
  sits in the junior row and beats every later row. "Assistant Manager, Finance"
  is `junior` too, while "Vice President - Quant" and "Associate, Investment
  Banking" name no level. A senior-grade role can therefore pass the level rule.
  That is a false pass (noise in the digest), not a false hide.

## Evidence
- `rolescan/scoring/facts.py`: `resolve_level` (its step list and the `source`
  overwrite), `_LEVEL_WORDS`, `level_from_text`, `_FIELD_WORDS`, `resolve_field`.
- `rolescan/scoring/rules.py`: the `level_from_title_only` branch in `_rule_skip`.
- `rolescan/config.py`: `RulesConfig.level_from_title_only`.
- `tests/test_level.py`, `tests/test_level_title_only.py`, `tests/test_field.py`,
  `tests/test_field_rule.py`.
