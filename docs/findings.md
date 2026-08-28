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

### The rate card was wrong for eight of ten entries, and said so

Every match reported roughly **half** its true cost. `llm-run-2` was billed as
$1.18 and actually cost $2.36. Which means the $75 safety halt - the one number
the whole budget system exists to enforce - was in practice a $150 halt.

| model | card | vendor |
|---|---|---|
| `grok-4.3` | 0.20 / 0.50 | **1.25 / 2.50** |
| `gpt-5.6` | 1.25 / 10.00 | **5.00 / 30.00** |
| `gpt-5.4-mini` | 0.25 / 2.00 | **0.75 / 4.50** |
| `gemini-3.6-flash` | 0.30 / 2.50 | **0.75 / 3.75** |
| `gemini-3.5-flash-lite` | 0.10 / 0.40 | **0.30 / 2.50** |
| `claude-sonnet-5` | 3.00 / 15.00 | **2.00 / 10.00** |

**The file predicted this exactly.** It carried a comment saying most of its
rates were the figures the roster was costed against and "should be re-checked
before a flagship run". That was accurate, and it was ignored - by me, while
quoting the numbers to two decimal places in a roster recommendation built on
them. A warning that names its own failure mode is not a control.

So provenance is per entry now: each rate cites the vendor URL it came from and
the date it was read, `make prices` prints the card for re-checking, and the
suite fails once any entry passes 90 days. The two entries the old card got
right were exactly the two its comment vouched for, which is the whole argument
for the change.

`rebuild_bundle` also stopped trusting the stored dollar figure. Tokens are a
measurement and cannot go stale; money is derived. Correcting a rate now
reprices history by rebuilding, instead of carrying the error into an artifact
somebody reads later.

**Two things worth knowing beyond the numbers.** Sonnet 5 was wrong by being too
*high* - its introductory pricing was made permanent - so rates rot in both
directions. And Claude 4.7+ models use a tokenizer producing ~30% more tokens
for the same text, which means per-token price comparisons understate them by
about a third. No sticker-price table settles a cost question; only measured
spend does.

*Postscript on how this surfaced.* Another model challenged the table and was
right that it was wrong, and right on three of four prices - while being wrong
about the cause (it argued a decimal shift; the arithmetic was correct on stale
inputs) and wrong about its own vendor's model, quoting flash-lite at
$0.54/$4.50 against a real $0.30/$2.50. Both of us were confidently wrong about
something. The vendor pages settled it in four fetches.

### The projection that clears a run for take-off was half of true

The rate card's sibling, and the same bug one file over.
`preflight.py` answers "can this account pay for a 300-turn match", and the
answer it gave was **$21.88** against a measured **$46.80**.

Its projection was five constants: 2k input, 4k output, 6.5k cached per turn,
multiplied by the turn count.
Both halves were wrong, in the same direction.

**One output figure cannot cover four seats.**
`gpt-5.4-mini` emits about 9,200 output tokens a call and `gemini-3.6-flash`
about 1,270 - seven times apart.
A single 4,000 charges the cheap seat too much and the expensive seat, which was
**57% of the entire bill**, less than half of what it costs.

**The prompt is a ramp, not a constant.**
The observation carries the board, so it grows as the board fills.

| seat | prompt, turn 1-20 | prompt, turn 121-128 | growth | R² |
|---|---|---|---|---|
| `gpt-5.4-mini` | 8k | 30k | 200/turn | 0.997 |
| `gemini-3.6-flash` | 6k | 26k | 144/turn | 0.953 |
| `grok-4.3` | 7k | 23k | 134/turn | 0.976 |

Multiplying a flat profile by 300 bills the turn-one board three hundred times,
and the error compounds exactly where the risk is: late in a long run, which is
the only stretch a 128-turn match never reached.

Measurements now live in `arena_orchestrator.profiles`, each carrying the match
it came from, the calls behind it and the fit quality, re-derivable with
`make profiles`.
The projection prints a range rather than a number - `output_high` is the worst
per-call mean any completed match has shown - because the question is not what a
run will probably cost but whether the account survives it going badly.
For this roster that range is **$46.80 to $62.14**, against a $75 halt.

**The one that nearly got away.** `claude-haiku-4-5`'s growth fits at R² 0.53 in
`baseline-300`, because it was eliminated on turn 35 holding one city and its
observation never grew. Fitted anyway, it gives 61 tokens a turn against the
142 the same seat shows across 51 turns of `shakeout-300` - a **40% under-projection**
of that seat, produced by arithmetic that was entirely correct.
A fit that low is not a noisy measurement, it is the absence of one, so
`make profiles` flags it and the suite refuses any entry below R² 0.85.

**What generalises.** The rate card was wrong about the price of a token; this
was wrong about the number of tokens. Both were internally consistent, both
produced plausible dollars, and neither could be caught by anything that did not
compare the number against a journal from a match that actually happened. That
comparison is now a test, skipped when no journal is present and binding when
one is.

### A dead seat is not a slow seat, and the loop could not tell

A 300-turn baseline reached turn 40, one seat's API credits ran out, and it
played **29 further turns with that civ holding its cities and issuing no
orders** - on its way to producing a four-way comparison missing a fourth. Six
hours of wall clock and about eighteen more dollars, had it finished.

Every piece of software involved behaved exactly as designed. "An agent that
cannot answer passes its turn, and the match continues" is the right policy for
a vendor having a bad ten minutes, and precisely the wrong one for an account
that cannot pay: that does not recover inside the run, so the fallback quietly
converts a billing problem into a corrupted result. The circuit breaker even
opened, correctly, and the match went on regardless.

Two changes, and the second matters more than the first:

- **`make preflight`** probes every seat with one tiny call before anything
  unattended, and `run_match` runs it automatically. It probes rather than
  reading a balance because no vendor here exposes one on a normal key - but
  every one of them will tell you immediately that you cannot spend, which is
  the answerable version of the question. It also prints the projected spend per
  seat, so the number you compare against a billing page is not a guess.
- **`OutOfCredits` halts the match**, the way the budget cap does: on a coherent
  board, scored, with `reason: "provider_credits"`. One lost turn instead of two
  hundred and sixty. It is matched on message text, because not one of the four
  vendors gives this its own status code or error type - it arrives as a 400 or
  a 429 whose body happens to mention money.

The general lesson is about fallbacks rather than billing. **A graceful
degradation that cannot distinguish transient from permanent will eventually
degrade gracefully for a very long time.** Tests now pin both directions: an
outage must not end a match, and an exhausted account must.

*Aside:* the preflight's first version had no retry, so a single transient stall
failed it on an account that was fine - the check was less resilient than the
thing it checked. It uses the same retry ladder now.

### The halt that saved the match made it unrecoverable

The sequel to the entry above, and a better example of the pattern than either
half is alone.
Stopping on `OutOfCredits` was right.
Recording that stop as `match_ended` was not, because that is the same record a
match writes when somebody **wins**.

`recover()` read `match_ended` and set `ended=True`; `run(resume=True)` saw
`ended` and refused with *"match at ... already ended; nothing to resume"*.
So topping up the account did nothing.
A seat running dry on turn 240 of 300 meant re-running the whole match, and the
safeguard added to stop a billing problem corrupting a run was now the reason a
billing problem destroyed one.

**Nothing was lost except the permission to continue.**
The journal is flushed per record, so every resolved turn was on disk, complete,
hash-verified and replayable.
The data survived; the flag on top of it said the match was over.

Reproduced end to end before it was fixed - halt on turn 5, top up, resume:

```
RUN 1 stopped: reason='provider_credits' at turn 5, spent $0.0467
journal: 5 turns resolved, ended=True
RESUME REFUSED: RuntimeError: ... already ended; nothing to resume
```

A process killed at the same point resumed perfectly, because a `kill -9` leaves
no ending record at all.
**The tidier failure was the unrecoverable one**, which is worth sitting with:
the loop stopping cleanly and writing down why produced a worse outcome than the
loop being shot in the head.

`match_ended` now distinguishes an **outcome** - conquest, domination, science,
turn limit, and final - from a **halt**: `provider_credits` or `budget_cap`,
where the board is coherent and scoreable but the match is not over and the
condition that stopped it is one a human clears with a billing page or a flag.
`HALTS` names the second set, `Recovered.resumable` answers the question
directly, and `run_match` prints the exact resume command when it halts, because
the run this protects is unattended and whoever reads that scrollback hours later
should not have to deduce that resuming is possible.

Two things fell out of the fix that were wrong on their own:

- **A resumed match reported $0.00 for every turn before the interruption.**
  Bundle frames carry per-seat spend, and replay rebuilt them from recorded
  decisions without the recorded costs. The tokens were in the journal the whole
  time; nothing read them on that path. `rebuild_bundle` did read them, so the
  data was always recoverable - the live artifact was just quietly wrong until
  somebody rebuilt it, which is the worst version of a bug to ship.
- **`rebuild_bundle` took the first `match_ended`, not the last.** A match that
  halted, was topped up and then finished carries two, and reading the earlier
  one publishes a completed run as having died of a billing problem it recovered
  from. `next()` over a journal finds the earliest match, which is the right
  answer for `match_created` and the wrong one here.

- **The spend column did not add up to its own total.** Resume carried
  `spent_usd` forward - so the cap, the safety-critical number, was right - and
  seeded every *seat's* dollars at zero. A resumed match printed a per-seat table
  summing to half the total above it, and `score_per_100k`, the efficiency figure
  this experiment exists to produce, divided a whole match's score by the spend of
  only the part played since the last interruption. Found by running the CLI and
  reading the output, not by a test: the two numbers were on screen together and
  disagreed, and nothing was comparing them.

**What generalises.** Two records that look alike are not alike, and the cost of
conflating them shows up only in the recovery path - the one place nobody
exercises until the day it matters. The halt had tests, thorough ones, in both
directions. Not one of them asked what happened *next*.

And the three defects behind it were all the same defect: state that resume
carried *partially*. The total but not the seats, the decisions but not their
costs, the record but not its meaning. Half-restored state is worse than
none, because the half that is present is what makes the result look plausible.

### Trading order enforcement for reasoning eliminated a civilisation

The most expensive mistake in the project, and it was mine, in a change I made
with the risk written into the comment beside it.

`claude-haiku-4-5` cannot have both extended thinking and the strict action
schema. I resolved that by loosening `orders` so only `action` was required and
demoting the field requirements to prose. The next 128-turn match:

| seat | orders sent | malformed | dropped |
|---|---|---|---|
| `claude-haiku-4-5` | 116 | 106 | **91%** |
| `gpt-5.4-mini` | 1037 | 0 | 0% |
| `gemini-3.6-flash` | 1636 | 5 | 0% |
| `grok-4.3` | 5570 | 0 | 0% |

It answered `{"action": "found_city"}` with no unit and no name, over and over,
across every order type - 35 malformed `move_unit`, 20 `found_city`, 17
`set_research`. Those entries are unusable, so `actions.parse` drops them. It
could not found cities, lost the one it had, **and was eliminated on turn 35**,
leaving a three-way match for the remaining 93 turns.

The prose was present and correct - `found_city requires name, unit_id` - and
the model ignored it 91% of the time. **Structured output works because the
decoder enforces the shape.** A description is a suggestion.

I shipped it on one live probe that happened to return well-formed orders,
without measuring the rate. `dialects.py` already said, in a comment I wrote
while making the change, that a loose union "lets a model answer
`{"action": "found_city"}` with no unit and no name - measured, repeatedly".

**The fix is to give up the other thing.** Pre-4.6 Anthropic keeps the strict
schema and does without extended thinking: 0 malformed across three live calls,
against 91%. A seat that reasons but cannot act is worth nothing; a seat that
acts without a visible trace is worth a great deal.

### The grammar ceiling is total complexity, not bytes

Worth recording because it rules out the obvious fixes. Chasing a way to keep
both, measured live against `claude-haiku-4-5` with thinking on:

| schema | bytes | union-typed params | |
|---|---|---|---|
| strict | 4,136 | 10 | rejected |
| strict, 3 fields removed | 3,950 | 7 | rejected |
| strict, 6 fields removed | 3,802 | 4 | **accepted** |
| one array per action type | 5,199 | 0 | rejected |
| loose orders | 4,924 | 0 in orders | **accepted** |

So shrinking bytes does not help - `{"type": ["string","null"]}` is still a
union - and eliminating unions does not help either if the structure grows to
compensate. The ceiling is the compiled grammar's total size, and no
reformulation found gets the strict schema under it with thinking enabled.

Two byte savings were kept anyway, because they are free headroom for every
Anthropic seat: **830 bytes of `title`** that Pydantic emits and a grammar never
reads, and the compact nullable encoding at 19 bytes a field. Together 1,211
bytes, from 5,347 down to 4,136. Neither solves the ceiling; both were always
removable and nobody had looked, because the schema had been small enough for
every model that mattered at the time.

### Turning on thinking shrinks the grammar budget

The third distinct Anthropic schema limit here, and the one that nearly cost a
seat. `claude-haiku-4-5` rejected the action schema with "compiled grammar is
too large" whenever extended thinking was enabled - and accepted the identical
bytes with thinking off.

Bisected live:

| | with thinking |
|---|---|
| strict dialect, 5,347B | 400 |
| loose `orders`, 4,924B | **works** |
| tiny schema, ~200B | works |

Loosening `orders` costs 423 bytes and buys the whole seat: valid orders and a
3,072-character reasoning trace where there had been none. `claude-opus-5` takes
adaptive thinking and the strict schema unchanged, so the dialect is model-aware
rather than provider-aware and the flagship keeps the stronger guarantee.

This was one measurement away from being resolved the expensive way. The
alternatives on the table were a reasoning-disabled civ in a four-way comparison
or swapping in a model at three times the price - and the roster argument had
already been made on both. **Bisect the limit before redesigning around it.**

### One seat counted its cached tokens twice

`Usage.input_tokens` is defined as the *uncached* input, because the pricer
charges `input * 1.0 + cache_read * 0.1` and folding them together would hide
the one number worth watching. Three adapters honoured that. The OpenAI
Responses adapter passed `usage.input_tokens` through unchanged, and on that
surface it is a total that already includes the cached portion.

So every cached token on that seat was billed at **1.1x instead of 0.1x** -
eleven times over on the cached share. Measured on a 19-turn shakeout: 65,280
of 133,185 input tokens were cached, and the seat was overcharged $0.016 of
$0.62. Small, because that model's cost is dominated by 292k output tokens.
On a flagship roster with a larger prefix and less output it scales the wrong
way.

The louder symptom was the cache-rate column, where **four seats were not
measuring the same quantity** - one reporting cached/total, three reporting
cached/uncached. That column is the tripwire for the 12.5x breakpoint bug
above, so it being quietly incomparable across vendors mattered more than the
cent and a half.

Two things had already stated the rule and neither enforced it: the `Usage`
docstring, and a comment in the sibling function forty lines below the bug
saying exactly why the subtraction is necessary. **A comment is not a test.**
There is a test now, and it asserts all four surfaces at once.

While fixing it: the viewer's cache column divided by *all* tokens including
output, so a seat reading half its prefix from cache displayed 13% purely
because it thought at length. Output tokens were never cacheable; the
denominator is input.

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

## The diplomacy console was lying, not empty

The viewer said "Nobody has spoken" for a match in which all four civs
negotiated non-aggression pacts, renewed them, and honoured them to the end.

The bundle gathered `message_sent` events and nothing else. But a model that
wants a pact does not send a message and then propose - it proposes, and puts
the diplomacy in the proposal's covering note. Of twenty utterances in a
19-turn match, **seven were invisible**, and they were the load-bearing ones:
both pact openings and every single acceptance. The replies were worse than
invisible - the engine never stored them at all, so `respond_to_proposal`'s
`message` was parsed, validated, and dropped on the floor.

This is a different failure from an empty panel. An empty panel says "I have
nothing"; this one asserted something false about the match, and did it in the
one place a reader would go to check.

A frame now tags each utterance `message`, `proposal` or `reply`, and the
viewer labels the three differently, because they are three different acts:
talk, talk with binding terms attached, and the answer that signs them or does
not.

### A model can sign a treaty without saying anything

`respond_to_proposal.message` is optional, and one model used that: haiku
accepted a ten-turn non-aggression pact with `message: null`. Every other
acceptance in two matches came with prose, so the silent one looked like a
missing feature - a pact appeared in the relations bar with nothing in the
thread to account for it.

It was not a missing feature and not an engine bug. It was a modelling mistake
of mine: wordless replies were suppressed on the reasoning that an empty chat
bubble is worse than none. True for a message, wrong for a treaty. In a
negotiation the *act* is the content, and a civ that bound itself for ten turns
without comment has done something worth seeing - arguably more interesting
than the ones that explain themselves. The viewer says "signed without comment"
now, and the reply's own reasoning that turn ("Accept p4 non-aggression pact to
secure western border") is right there beside it.

The general rule this produced: a `send_message` needs words because there the
words are the act; a proposal or a reply does not.

Two adjacent facts, both of which look like bugs and are not:

- **Every message in the match was private. There were no public broadcasts at
  all.** Not a rendering gap - the models simply never used the public channel.
- **There was no combat in nineteen turns.** Four civs, mutual pacts, and not
  one fight, so the combat panel is empty because the match was.

---

## Every match bought its models' reasoning and threw it away

Reasoning tokens are billed as output. They were counted, priced, and paid for
on every turn of every match - and the trace behind them was parsed out of the
response by the adapter and then dropped, because nothing downstream stored it.
The Anthropic adapter's own docstring said the text "matters more here than in
most applications, because reading the model's reasoning is the entire
experiment", four lines above the code that discarded it.

Traces now go to `transcripts.jsonl`, and their size to the journal so coverage
is auditable without opening eighty payloads. Not into the published bundle, for
the same reason the prompts are not: a published match should carry the match
and nothing else.

**What each vendor will actually give you**, measured live rather than assumed:

| Vendor | Trace | Notes |
|---|---|---|
| Anthropic | 367 chars on `claude-opus-5` | Needs `thinking.display`. **Nothing on `claude-haiku-4-5`** - adaptive thinking is 4.6-and-later, so the shakeout roster has no trace at all |
| OpenAI | 1,948 chars | Needs `reasoning.summary` **requested**. Without it the reasoning items arrive with an empty summary list and 774 reasoning tokens are billed for nothing retained |
| xAI | 132 chars | `reasoning_content` on the message - a vendor extension to a shared schema, absent from the SDK's typed model, so read by name |
| Google | **none** | Billed 1,137 thought tokens and exposes no thought text on the interactions surface |

So an absent trace means "this vendor did not offer one", never "the model did
not think". Two of four seats on the shakeout roster produce nothing, and one of
those still charges for it.

Worth keeping distinct from the `reasoning` block the action schema requires.
That block is the account a model writes *knowing it will be read and handed
back*; the trace is the deliberation behind it. They are different evidence, and
the interesting comparison is between them.

---

## Viewer bugs, which fail the same way

Both of these drew nothing and said nothing. A panel that throws leaves a
stack trace; a panel that is never reached leaves an empty box that reads as
"no data this turn", which is a plausible answer in a game viewer and so does
not prompt anyone to look.

- **A `return` in a nested branch returned from the whole render.** The
  dossier's "all civs" view ended in a bare `return`, written to mean "done
  with the dossier" and actually meaning "done with `panels()`". Everything
  after it - Events, Signals, every count badge - stopped rendering. "All" is
  the *default* view, so this was the normal case, not an edge one. Found by
  staging the viewer against a free bot match before pointing it at a paid
  bundle, which is now the habit.
- **A block placed above the values it read** threw a temporal-dead-zone error
  on every render, and the page absorbed it by not drawing the sections below.
- **A column with no room is a column that says nothing.**
  Replacing `p1..p4` with real model ids made the cost table unreadable: six
  numeric columns took 273 of the panel's 331 pixels and left the model name
  59, so every row identified itself as `cl...`.
  Two rows both truncated to `g...`.
  The name also had no ellipsis, because the flex wrapper inside the cell
  swallowed the one on the cell itself - so it clipped mid-glyph and did not
  even look truncated.
  The fix was to shorten the *headers*, which each already carry a full
  sentence on hover, rather than the data.
  Worth stating because the reflex is to shorten the identifier, which is the
  one thing on the row that cannot be recovered from context.
- **Two boards at once, and only a moving label revealed it.** `drawTurn`
  cleared the scene, then awaited the fold, then drew. A scrub during the fold
  started a second draw that cleared and drew correctly - and then the first one
  resumed and added its stale geometry behind it.
  The duplicates coincide wherever nothing moved, so a city on the same tile in
  both turns drew twice in the same place and looked like one city.
  It survived every previous panel because everything drawn was either static
  or overlapping. It became visible the moment a nameplate was planted on a
  wandering army, which is somewhere different every turn: the board showed the
  same empire twice, in two places, one of them several turns out of date.
  Fixed with a sequence number checked after the await. The lesson is about
  where to look rather than about renderers - the bug was years-old in project
  time and was found by adding a feature that happened to move.
- **The wilderness got a nameplate.** It is modelled as a player so combat and
  movement work unchanged, so it owns units, so a label keyed on unit ownership
  named it. The engine's own test file opens by warning about exactly this and
  the viewer made the mistake anyway, which says the warning needs to be a
  helper rather than a paragraph.

The first two suggest a rule: a `return` is only trustworthy in a function that
does one thing. Both blocks are their own functions now.

The last two suggest another, and it is the more useful one. Three of the five
were invisible until something *moved* - a label on a wandering army, a name on
a roaming wolf pack, a column that had to hold a real string. A viewer tested
only against a settled board tests almost nothing.

---

## A pytest marker that does not exist excludes nothing

`pyproject.toml` sets `addopts = "-m 'not contract'"` so the paid contract
tests stay out of the default run.
Passing `-m "not live"` on the command line - a marker this project has never
defined - **overrode that**, because a later `-m` replaces the earlier one, and
`not live` is true of every test in the suite.
The command that was meant to be *more* cautious than the default selected the
live provider tests instead.

Nothing failed loudly: the contract tests skip themselves when a key is absent,
and the keys are present here, so the only signal would have been the bill.

The rule: never pass `-m` to this suite.
The default already excludes what costs money, and any `-m` silently discards
that default.
`make contracts` is the deliberate way in.

---

## Three quarters of the violence never reached the viewer

Every panel that counts fighting was counting a quarter of it.

A civ's attack emits `combat_resolved` carrying `defender_owner`, both unit
types and both death flags.
The wilderness emits the same event type from `barbarians._strike` with none of
those fields.
The bundle's combat feed keys on `attacker_type`, so it silently kept the civ
attacks and dropped every wilderness one: **28 of 103 blows struck in the first
complete match**.

The consequence was not a missing panel but a confident wrong one.
A civ mauled by wolves thirty times rendered identically to a civ at peace,
and the Signals combat bar - whose whole job is "who is being worn down" -
answered with the subset of fights that happened to carry a type field.

It also inverted a headline.
Counting only what the bundle carried, the match looked like 20 fights against
the wilderness and 8 between civs, a ratio of 2.5.
The true figure is 95 and 8, a ratio of 11.9 - the same finding, four times
understated, and it was about to go into a public post at the wrong number.

The event payload is now identical on both paths, and a test asserts each field
by name rather than asserting that some payload exists - the bug was an
omission, and a shape check would have passed against it.

Bundles built before the fix cannot be repaired, because the journal never had
the fields.
The viewer therefore reads wilderness-initiated fights from the events for those
matches, and skips the fallback when it sees a bundle whose `combat` already
contains a wilderness attacker.

### Treaty paperwork was being counted as conversation

Smaller, same shape.
`frame.messages` carries three kinds - `message`, `proposal`, `reply` - and the
fold counted all three as things a civ said.
That reported 585 private messages against a real 481, and made a civ that
negotiated a lot look like a civ that talked a lot.
Proposals are counted where they mean something, which is the diplomacy tally.

The rule both share: when one producer writes a record and another reads it by
looking for a field, the reader silently defines what counts.
Neither of these threw, logged, or drew an empty box.

---

## Costs

Measured, not estimated. Shakeout roster, seed 4, a complete 20-turn match on
the corrected accounting - the first run where all four cache figures are the
same quantity (cached over total input).

| Seat | Model | $/turn | Median latency | Cache read | Output tokens | Score |
|---|---|---|---|---|---|---|
| p1 | `claude-haiku-4-5` | $0.0142 | 23s | 50% | 36k | 48 |
| p2 | `gpt-5.4-mini` | $0.0388 | 120s | 58% | 379k | 94 |
| p3 | `gemini-3.6-flash` | $0.0044 | 35s | 0% | 20k | **101** |
| p4 | `grok-4.3` | $0.0014 | 9s | 22% | 10k | 71 |

**$0.059 per turn for the whole table.** 300 turns extrapolates to ~$18 on this
roster. Flagship models are several times that and still sit inside the $75 cap.

The result that makes the case for measuring cost per point at all: **the seat
that won spent a twentieth of what the second-place seat spent.** Gemini took
101 at $0.087 while gpt-5.4-mini took 94 at $0.777, having produced *nineteen
times* as many output tokens. Cheapest per point was grok at 46 points per
100k tokens against gpt's 18.

One match on one seed proves nothing about model quality, and the caveat below
about lost turns matters before anyone reads a ranking into it. But it is
exactly the shape of finding a leaderboard by score alone would hide.

**One seat is 65% of the bill and ten times the latency of another.**
`gpt-5.4-mini` produced 293k output tokens against grok's 11k - twenty-six
times as many - for a middling score. Turns resolve simultaneously, so it also
sets the wall clock for everyone. Whether that thinking buys anything is the
question this lab exists to answer, and it is now measurable per seat.

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

### A timeout costs two turns, not one, and cannot be retried

Two complete 20-turn matches lost agent-turns to the 420s deadline. The second
run is the alarming one: **p1 timed out on turns 13 and 14 back to back**, then
missed the cache on 15, 19 and 20. Its score came in last at 48, against 71 for
the same model in the first match.

That is the failure this limit exists to prevent, arriving through the limit
itself. **p1's median latency is 23 seconds.** A 420s timeout on that seat is
not a model thinking, it is a stalled request - and the seat is being scored
down for a transport problem rather than for how it played, which is precisely
the vendor bias the whole design is trying to avoid.

Two structural problems behind it:

- **The retry has nowhere to run.** Each SDK client is given a 400s timeout,
  deliberately just under the 420s turn deadline so a hang is classified as a
  transport timeout rather than an opaque turn failure. But that leaves 20
  seconds, which is not enough to place a second call. The classification works
  and the recovery it was meant to enable does not - no `provider_retry` was
  journalled for either lost turn.
- **The next turn pays too.** A 420s timeout is seven minutes, and Anthropic's
  cache TTL is five. So the turn after a timeout necessarily finds a cold
  prefix and writes a fresh entry at 1.25x instead of reading at 0.1x. The
  journal shows exactly that: `agent_failure` on turn 7, `cache_miss` on turn 8.

The fix was not a smaller SDK timeout. 200s would leave room for a real retry,
and would also cut off a seat whose median is 120s and whose calls run to
~19,000 output tokens - manufacturing the vendor bias the 420s exists to
prevent. Every value was wrong for one case or the other, because the two limits
were answering different questions through one number.

**Fixed by measuring silence instead of duration.** A model streaming tokens is
alive however long it takes; a stream that has produced nothing for ninety
seconds is not going to speak again. So `stall_gap_s` resets on every event: a
five-minute thinker costs nothing, and a dead socket is caught in ninety
seconds, which finally leaves room inside the turn for the retry a hang needs.

- `stall_gap_s` (90s) catches hangs. Raises `Stalled`, a subclass of `Timeout`,
  so the existing retry ladder picks it up without knowing it exists while the
  journal can still tell a hang from a model that thought too long.
- `turn_timeout_s` rose 420s -> 600s and became what it should always have been:
  a backstop wide enough for one full attempt plus one retry.
- The guard wraps the iterator rather than rebuilding streaming around it, so
  each adapter's usage, refusal and stop-reason handling is untouched - the parts
  most likely to break.

**Two seats, not four.** Anthropic already streamed; OpenAI now does, for the
guard rather than for the tokens. xAI's chat-completions surface could and does
not yet. Google's interactions surface exposes no token stream at all, so that
seat still relies on the transport timeout and the backstop. Recorded here
rather than papered over: the policy is not yet uniform across the table, and a
four-way comparison with two different hang policies is a smaller bias than the
one it replaced, not zero.

All ten live contract tests pass on the streaming path, which matters more than
usual here - this is precisely the class of change that passes offline and fails
on the wire.

Same reason a resume always pays a cache write on its first turn: any gap longer
than five minutes is a cold prefix. Worth knowing before reading a resumed run's
cache column and concluding something broke.

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
4b. `make prices` before quoting any cost. Rates rot in both directions, and a
   comment saying "these should be re-checked" is not a control - it was there,
   it was accurate, and the numbers were quoted anyway.
4c. When a vendor limit blocks a design, **bisect it before redesigning around
   it**. The grammar ceiling looked like it cost a seat its reasoning; it cost
   423 bytes.
4c-bis. **Never trade schema enforcement for anything.** It is the only thing
   standing between a model and an unusable order, and the failure is silent -
   dropped orders look exactly like a model choosing to do nothing. Measure the
   malformed rate over a real match before believing a looser schema is safe.
4d. `make preflight` before anything unattended. A key that exists is not a key
   that can pay, and the gap between those two facts cost six hours.
5. Stage the viewer against a bot match before a paid one. It costs nothing and
   it is the only way to tell "this match had no diplomacy" from "this panel
   never rendered".
