# Findings

What went wrong on the way here, what it cost, and what to check first next time.

Kept because most of these were invisible until something live disagreed with
something offline, and the same class of failure will happen again the next time
a vendor changes an API or someone adds a field to the action schema.

---

## The recurring shape of every real bug

Almost nothing here failed loudly.

Of the fourteen live-only bugs below, **one** raised an exception at the point of
the mistake. The rest returned HTTP 200 and were wrong: a parameter silently
dropped, a token counter reading zero, a cache written and never read, an order
quietly discarded. The offline test suite was green throughout.

The lesson is not "write more tests". It is that a test which never sends the
real bytes to the real endpoint cannot see this class of failure at all, and
three separate layers were needed before they stopped hiding:

| Layer | Cost | Catches |
|---|---|---|
| Mocked adapter tests | free, every commit | our own parsing and error classification |
| **SDK surface tests** | free, every commit | a vendor renaming a method or field |
| Live contract tests | cents, on demand | a vendor changing what it accepts |
| **Runtime journal checks** | free, during a match | what all three above still missed |

The two in bold did not exist until the bugs they catch had already happened.

---

## Vendor API bugs

Every one of these passed the full offline suite.

### Model ids are the most fragile thing in the project

Four ids were hardcoded and **none of them existed**:

| Written | Reality |
|---|---|
| `gemini-3.6-pro` | 404. There is no Gemini *pro* above 3.1, and that is a preview |
| `grok-4` | Does not exist. The live list is 4.3 / 4.5 / 4.6 |
| `gpt-5.6-mini` | Does not exist. The newest mini is 5.4 |
| `gemini-3.6-flash` | This one was right |

Two reported the failure as something else entirely. xAI answers an unknown
model with `400 Incorrect API key provided`, which sends you to the billing page.

Model ids now live in `.env` (`ARENA_FLAGSHIP_GOOGLE` and friends).

### xAI authenticated with the OpenAI key

`XAIClient` rides OpenAI's SDK, whose key fallback is `OPENAI_API_KEY`. With no
key passed it sent the *OpenAI* credential to `api.x.ai`, which replied
`Incorrect API key provided` - entirely correctly. That message cost an hour
spent on billing and model ids before the code was suspected.

### Google was wrong in seven places

Written from a documentation summary, mocked against the same misunderstanding,
green the whole time.

- `instructions=` should be `system_instruction=`. The SDK takes `**body`, so a
  misspelled parameter is **not an error, it is an omission** - the entire rules
  reference would have gone nowhere and an agent would have played a multi-day
  match with no rules, at full price, with nothing in any log to explain it.
- `response_format` is a *list* of per-modality formats, not a dict.
- Each entry must be a plain dict; the SDK's own `TextResponseFormat` fails to
  unmarshal.
- The schema key is `schema` on the wire although the Python field is called
  `jsonSchema`. Sending `jsonSchema` returns 200 and is **silently ignored** -
  the model answers in prose and every turn fails to parse.
- `max_output_tokens` lives inside `generation_config`.
- There is no `finish_reason` and no `candidates`; completion is `status` and a
  refusal is a status plus an `errors` list.
- Usage counters are `total_*`. The `candidates_token_count` name belongs to the
  old `generate_content` response, so **every Gemini turn priced its output at
  zero** and the budget halt was disarmed on that seat.

Introspecting the installed SDK caught six of these for free. It also gave false
confidence on the seventh: `jsonSchema` *is* the model field, so asserting it
existed passed while the wire wanted the alias. **Introspection tells you what a
name is called locally, not what goes on the wire.**

### Reasoning tokens bill against the output cap

Not on top of it. An 8k budget was consumed entirely by thinking and the
response came back `incomplete` and empty. Both Anthropic and OpenAI caps are
now 32k; unused headroom is free, since billing is on tokens produced.

The adapter reported this as `empty response body` because it checked content
before status - true, useless, and it sent the investigation back to the schema.
**Check status before content.**

### Adaptive thinking is 4.6-and-later

Sending it to `claude-haiku-4-5` is a 400, not a downgrade. The shakeout roster
lost every turn while the flagship roster was fine.

---

## Schema bugs

### `oneOf` is rejected by Anthropic and OpenAI

Pydantic emits it for discriminated unions. `anyOf` means the same thing here
and is accepted everywhere.

This is the one to remember, because of *how* it hid: every mocked test passed,
the schema parity test passed, and a 108-turn dry match played through without
complaint. **None of those ever send the schema to a vendor.** It would have
400'd on turn one of the flagship run, on three of four seats at once.

### A `$ref` may not carry a `description`

Adding descriptions to the eight dossier fields - the fix for two models never
writing an opponent model - broke OpenAI on turn one of the next run:

```
$ref cannot have keywords {'description'}
```

`trustworthiness` is an enum, so Pydantic renders it as a `$ref` to a `$defs`
entry, and a `Field(description=...)` becomes a sibling of that `$ref`. In
draft-07 a `$ref` *replaces* its whole node, so a sibling is ambiguous by
construction and OpenAI rejects it rather than ignoring it. Only that one field
of the eight was affected, which is why nothing looked wrong on inspection.

The sanitizer drops the siblings now. The description is not hoisted into an
`allOf`, which is the other legal fix, because that costs bytes Anthropic's
grammar cap does not have - and the same guidance already reaches every model
through the rules reference, in the cached prefix, at a tenth the price.

**The gap this exposes is the interesting part.** The schema was valid to us,
the parity test passed, `make contracts` passed, and the failure still arrived
on a paid live turn. The dialect test asserted about keywords we *strip*; it
had nothing to say about a keyword that is legal everywhere except beside a
`$ref`. It does now.

### Anthropic enforces two undocumented schema limits

Found by bisecting against the live API:

- a **compiled grammar size** cap, around 6KB
- a cap of **16 union-typed parameters**

The action schema exceeds the first. Flattening both unions clears it and trips
the second at 24. That leaves room for exactly one strictly-flattened union, so
`orders` is flattened strictly and `diplomacy` loosely - a bad order costs the
turn, a bad message costs one message.

The "one schema, all four vendors" premise in the design is **false**, and it is
false about the vendor it was authored for. Only Anthropic needs a dialect.

### The observation and the schema disagreed about action names

`legal_actions` offered `move`, `build` and `research` while the schema demanded
`move_unit`, `set_production` and `set_research`. Six of twelve action names
never appeared in an observation under the name the model had to emit, and
`set_rates` appeared nowhere at all.

A model reads what it is shown this turn before it consults a schema it was
handed once. Vendors that enforce the enum in the decoder made the contradiction
*impossible to act on and therefore impossible to see*. Gemini, which does not,
followed the observation and emitted `research` and `set_taxes`. Those orders
were dropped as invalid, so the only symptom was a civ quietly doing less each
turn than it had ordered - **on the one seat where a weaker model was the easier
explanation.**

I had attributed this to Gemini's loose enum enforcement for two days.

---

## Cost bugs

### The cache breakpoint was on the wrong block - 12.5x

`cache_control` at the top level caches the *last cacheable block*, and the last
block is the user turn, which carries the observation and differs every turn. So
the cache key included the board. Measured with nonces so neither run could hit
a leftover entry:

```
top-level        call 0: write=6846 read=0
                 call 1: write=6846 read=0
                 call 2: write=6846 read=0
on system block  call 0: write=5036 read=0
                 call 1: write=0    read=5036
                 call 2: write=0    read=5036
```

Writing at 1.25x every turn instead of reading at 0.1x. Nothing fails. The only
symptom is a larger bill.

**The contract test for this passed for a whole live match**, because a previous
run had left a matching entry inside the five-minute TTL. It uses a nonce now.
Even so it cannot discriminate the two placements through the streaming path -
three attempts failed - so placement is guarded by an offline assertion on the
request shape and by a runtime `cache_miss` journal record instead.

### Resume reset the budget meter

The cap belongs to the match, not to the process running it. A run that crashed
near its limit could resume and spend the whole cap again; a run that crashed
repeatedly had no limit at all. Found on a real interrupted shakeout: $1.54
already spent, resumed process reported $0.22.

### The system prompt sat in the publishable bundle

The run directory held `journal.jsonl`, `transcripts.jsonl` *and* the bundle, so
serving a match served the prompts with it. The bundle is a subdirectory now.

The test guarding this globbed `*.json`, and transcripts are written to
`transcripts.jsonl` - **the one test between a system prompt and a published web
page was not looking at the file the prompt was in.**

---

## Engine bugs worth remembering

Found by bot matches long before any money was spent, which is the argument for
having them.

- `ev.event()` took `text` both positionally and by keyword, so **every message
  send crashed the reducer**. Bots never talk, so no bot match caught it.
- Wonders could be built eight times - uniqueness was checked at queue time, not
  at completion.
- `buildable()` offered `wolf` and `barbarian` to civs. Cost 0, so instant.
- Bankruptcy did nothing: gold was clamped at 0 and the debt never applied. Only
  found by reading an actual observation payload showing `gold: 0,
  gold_per_turn: -19`.
- Seed 214 of 300 generated a **disconnected map**. Kept as a named regression.
- A bot built 161 warriors and 4 settlers, because a static priority list always
  re-picked the always-available option.
- `legal_actions` spelled out empty options for every unit: **5,400 tokens a
  turn**, more than the rest of the payload combined.

---

## Costs

Measured, not estimated. Shakeout roster, 30 turns, four live vendors.

| Seat | Model | $/turn | Median latency | Cache read |
|---|---|---|---|---|
| p1 | `claude-haiku-4-5` | $0.0127 | 19s | 52% |
| p2 | `gpt-5.4-mini` | $0.0269 | 82s | 28% |
| p3 | `gemini-3.6-flash` | $0.0030 | 25s | 0% |
| p4 | `grok-4.3` | $0.0012 | 6s | 14% |

**$0.044 per turn for the whole table.** 300 turns extrapolates to ~$13 on this
roster. Flagship models are several times that and still sit inside the $75 cap.

Two things that matter more than the totals:

- **The slowest seat sets the wall clock.** Turns resolve simultaneously, so a
  300-turn match runs at the pace of whichever model thinks longest. At 82s
  median that is roughly 7 hours.
- **Google reports no cache reads at all.** It caches implicitly on prefix
  match, and either the prefix is not stable enough to hit or the counter is not
  surfaced. Worth checking before a flagship run; it is the difference between
  paying 1x and 0.1x for the rules reference on that seat.

Timeout was raised 180s -> 420s: a 30-turn shakeout lost five of the last seven
turns on one seat at 107s median and ~15,500 output tokens a call. That is a
model thinking, not a hung request, and cutting it off manufactured exactly the
vendor bias the limit exists to avoid.

---

## Two things investigated and one of them deliberately not fixed

### The dossier cap did not cap what it claimed to

The design says "capped at roughly 2000 tokens". The implementation capped the
*number* of lessons and commitments at twelve, which is not a size cap: a model
writing twelve long ones sails past it. Measured on the first live match, one
agent's dossier reached 10,536 characters - about 2,600 tokens - while passing
the count check on every turn. The dossier is re-sent verbatim on every turn, so
an unbounded one is the easiest way for an agent to quietly triple its own input
bill.

It is a size budget now, trimmed cheapest-content-first: lessons, then
commitments. **Doctrine and opponent models are never trimmed** - the doctrine is
the plan being executed and the opponent models are the record this lab exists to
read. Mangling an assessment to save tokens would destroy the evidence to protect
the bill.

Separately: two of the four models wrote **no opponent models at all**, and the
field carried no description, so its name was the only guidance. All eight
dossier fields now have descriptions, and the rules reference explains the
dossier directly.

That created its own problem. Descriptions cost 1,400 bytes of schema, which
pushed the Anthropic dialect over its ~6KB grammar cap. So the dialect strips
them - before flattening, not after, because the flattening writes a description
of its own that is load-bearing - while the other three vendors keep them, and
the same guidance reaches everyone through the cached rules reference.

### Gemini's zero cache rate is real, expected, and not worth fixing

Implicit caching does not engage below roughly **17k tokens**. Measured: at a
2.2k-token prefix and a 7.3k-token prefix, zero cached on every call; at 17k, it
cached 8,174 tokens from the second call onward. Our system prefix is 2,234
tokens, so it does not qualify.

The Interactions API has no `cached_content` parameter either - only
`previous_interaction_id`, which chains the whole prior conversation forward and
is precisely the growing-context design this project rejected.

The arithmetic settles it:

| | per match (300 turns) |
|---|---|
| Gemini prefix uncached | $0.20 |
| If it cached | $0.02 |
| **Saving available** | **$0.18** |
| Cost of padding the prefix to qualify | $0.84 |

Padding to reach the threshold costs four times more than not caching at all.
The amber flag in the cost meter stays, because a zero on a vendor with an
explicit breakpoint means the 12.5x bug above - but it now carries a tooltip
saying which case is which.

---

## What to check first, next time

1. `make contracts` before any unattended spend. Nine tests, a few cents.
2. After the first two turns of a real run, read the journal for `cache_miss`,
   `provider_retry`, `throttled` and `agent_failure`. All four should be empty.
3. If a seat is scoring low, check whether its orders are being *dropped* before
   concluding the model is weak. That mistake cost two days.
4. If a vendor error names something that sounds like a credential or a billing
   problem, verify the model id first.
