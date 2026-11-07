# tender-screen: 2026-11-07-claude-haiku-5-5 (live)

Model `claude-haiku-5-5` via anthropic, 3 attempt(s) per case, 15 s, tokens {'input': 58557, 'output': 6164}.
pass@1 97%; pass^3 92% (11 of 12 conclusive cases pass every attempt; Wilson 95% 65%-99%). Inconclusive cases are outside these figures.
2 offline-only case(s) not run: they need their own answer.

| Split | Cases | Pass | Fail | Flaky | Inconclusive |
|---|---|---|---|---|---|
| visible | 7 | 6 | 0 | 1 | 0 |
| held-out | 5 | 5 | 0 | 0 | 0 |

| Tag | Cases | Pass | Fail | Flaky | Inconclusive |
|---|---|---|---|---|---|
| dates | 1 | 1 | 0 | 0 | 0 |
| lists | 2 | 2 | 0 | 0 | 0 |
| negation | 1 | 1 | 0 | 0 | 0 |
| numbers | 1 | 1 | 0 | 0 | 0 |
| quote-check | 1 | 1 | 0 | 0 | 0 |
| rules | 2 | 1 | 0 | 1 | 0 |

| Case | State | Why |
|---|---|---|
| tn-small | flaky | outcome=review (model's facts changed between attempts) |

## Held-out

Ids and states only, so nobody tunes to them.

- h-tn-1: pass
- h-tn-2: pass
- h-tn-3: pass
- h-tn-4: pass
- h-tn-5: pass
