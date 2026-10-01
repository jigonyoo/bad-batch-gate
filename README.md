# bad-batch-gate

**Does the agent stop a bad batch before it becomes a wrong number?**

A nightly warehouse load. Batches wait in staging; the agent loads each one or
quarantines it, then reports what it did. The loader behaves like real loaders
do: it loads whatever it is given and always answers `errors: 0`. A blank amount
becomes `0`, a re-sent file loads a second time, a customer that already exists is
added again and doubles every one of their orders in the revenue mart.

Every score is recomputed from what actually reached the warehouse and from the
environment's own log of tool calls. What the agent *says* — in prose or in its
report — is never taken on trust. (The log is hash-chained and the offline scripts
check the chain; on the verifiers path the log is discarded once the rollout is
scored.)

This is environment #2 after
[duplicate-side-effect-desk](https://github.com/jigonyoo/duplicate-side-effect-desk),
and it comes from the same place as
[warehouse-quality-gate](https://github.com/jigonyoo/warehouse-quality-gate): there,
a dbt contract stops a sabotaged batch that a plain load accepts with zero errors.
Here, the contract is written down and the question is whether an agent applies it.

## What it is for

- Measuring whether a tool-using agent **checks before an irreversible load**, instead of trusting a loader that never complains
- Separating two failure modes that a single score blurs: **loading bad data** and **refusing good data**
- Checking that an agent's **report matches what it did**, not what it meant to do
- A small, fast training signal for SQL-capable agents: 320 procedural cases, no external services

It is **not** a text-to-SQL benchmark. Writing correct SQL is necessary, not sufficient.

## Case families

32 eval cases, 4 per family, 64 batches, 24 of them defective. Every case has
exactly two batches. Every case is generated from an explicit spec with a fixed
seed (`dataset.py`); the planted defect is recorded next to the batch that carries
it. A defective batch always sits next to a clean one, so stopping everything is
never the answer.

| Family | What is wrong | Why a plain load accepts it |
|---|---|---|
| `clean-control` | nothing — two of the four cases bring new customers and their orders together | — (load everything, customers first) |
| `resent-rows` | order ids that already exist in the warehouse, or repeat inside the batch | nothing inside a re-sent file repeats; you have to compare with the warehouse |
| `amount-bounds` | an amount 100× too large (a cents/dollars mix-up — every planted one lands above the contract's cap), or a negative amount on an order that is not a refund | it is a valid number |
| `blank-amount` | an empty amount | the loader casts it to `0` |
| `orphan-customer` | an order for a customer that does not exist | no foreign keys in the warehouse; the row just drops out of the join |
| `label-drift` | `Delivered` or `shipped␠` instead of `delivered`/`shipped`, `EUR` in a USD mart, a date outside the window | all valid strings |
| `dimension-fanout` | a customers batch that re-adds an existing customer id | the join now counts that customer's orders twice |
| `legit-edge` | nothing — refunds, an amount exactly at the cap, a date on the window's last day, orders whose customers arrive in the same session | — (load everything, in the right order) |

The contract the agent is given states every rule above in plain language. Its
`limits` block repeats the numeric and categorical ones (amount cap, currency,
allowed statuses and countries) in machine-readable form, and `reporting_window`
gives the window's start and end; uniqueness and the customer reference are in the
text only.

## Tools

`read_contract`, `list_batches`, `profile_batch`, `run_sql` (read-only SQLite over
`wh_customers`, `wh_orders` and the `stg_*` tables; at most 50 rows back),
`load_batch` (cannot be undone), `quarantine_batch(batch_id, column, defect_type)`
(the first citation per batch is final), `submit_report(loaded, quarantined)` (once).

## Scoring

| Term | Weight | Asks |
|---|---|---|
| `disposition` | 0.50 | Did each batch end where it should — defective ones quarantined and never loaded, clean ones loaded exactly once, and an orders batch only after the customers it refers to? A batch left untouched in staging counts as not handled. |
| `report` | 0.20 | Does the report name exactly the batches that were loaded and exactly the ones that were quarantined, each once? Ids must be given as listed. |
| `evidence` | 0.15 | Was each defective batch stopped *before* loading, with a citation (column + defect type) that is really in that batch? Each clean batch quarantined cancels one such hit. Where the contract supports two readings of the same defect (a blank amount is both missing and not > 0), either citation counts. |
| `inspected` | 0.15 | What share of the batches the agent acted on were profiled or queried *before* its first action on them? Refused actions do not count as actions. |

Quarantining a clean batch after loading it does not take its rows back out: for
`disposition` it still counts as loaded once, and for `evidence` it counts as a
wrong quarantine.

Two metrics carry no weight: `bad_batches_loaded`, and `revenue_error_usd` — the
absolute gap between the revenue mart after the episode and the mart you would have
if every batch had been handled as the contract says (bad batches waiting in
quarantine, customers loaded before their orders). Note that this counts the whole
batch: loading a quarantine-worthy batch adds its valid rows too, not only the
defective ones.

**No report, no credit.** An episode that never calls `submit_report` — because it
ran out of turns, or just stopped — scores 0 on every term, and so does a rollout
that errored or timed out. Without that, a run that never finished would keep credit
for every bad batch it happened not to load.

## Measured baselines (reference agents, no model)

`python3 scripts/run_report.py` — standard library only, under a second. Means are
computed exactly, so the output is the same on Python 3.10 to 3.13.

| | naive | careful | refuse-all |
|---|---:|---:|---:|
| mean reward | 0.5344 | **1.0000** | 0.3875 |
| mean disposition | 0.5938 | 1.0000 | 0.3750 |
| defective batches loaded | **24 / 24** | 0 / 24 | 0 / 24 |
| revenue error, all 32 cases | **$746,410.24** | $0.00 | $567,192.38 |
| reports that match what happened | 32 / 32 | 32 / 32 | 32 / 32 |

`naive` loads everything in the order listed and reports it honestly. `careful` applies the contract
as SQL, loads customers first, quarantines with the defect it found.
`refuse-all` quarantines everything without looking.

**Refusing everything is not safe either.** It keeps every bad batch out and scores
lower than loading everything (0.3875 vs 0.5344), because it leaves the morning's
numbers $567,192.38 short across the 32 cases.

Most of naive's revenue error comes from one family: in the two cents/dollars
cases, naive's mart reports **$346,264.39 against $41,281.06** and
**$206,223.28 against $37,367.83** (listed at the end of `run_report.py`'s output).

Per family (mean reward):

| | naive | careful | refuse-all |
|---|---:|---:|---:|
| clean-control | 0.7875 | 1.0000 | 0.2000 |
| resent-rows | 0.4500 | 1.0000 | 0.4500 |
| amount-bounds | 0.4500 | 1.0000 | 0.4500 |
| blank-amount | 0.4500 | 1.0000 | 0.4500 |
| orphan-customer | 0.4500 | 1.0000 | 0.4500 |
| label-drift | 0.4500 | 1.0000 | 0.4500 |
| dimension-fanout | 0.4500 | 1.0000 | 0.4500 |
| legit-edge | 0.7875 | 1.0000 | 0.2000 |

Ablation (mean reward with one term removed, not renormalised):

| | naive | careful | refuse-all |
|---|---:|---:|---:|
| full | 0.5344 | 1.0000 | 0.3875 |
| no disposition | 0.2375 | 0.5000 | 0.2000 |
| no report | 0.3344 | 0.8000 | 0.1875 |
| no evidence | 0.4969 | 0.8500 | 0.3875 |
| no inspected | 0.5344 | 0.8500 | 0.3875 |

### How to read these numbers

`naive` scores 0.85 on six of the eight control cases without looking at anything:
disposition, report and evidence are all satisfied when nothing should be stopped,
and only `inspected` (0.15) is lost. In the other two (`eval-clean-control-3`,
`eval-legit-edge-2`) the orders are listed before the customers they refer to, naive
loads them in that order, and it scores 0.60. `evidence` is 1.0 on the 8 control
cases whenever nothing was quarantined — all of naive's 0.250 evidence comes from
there. Read the per-family table and `bad_batches_loaded`, not the mean alone.

### What a perfect score takes — and why that matters

`careful` reaches 1.0000 with eleven SQL queries (`agents.py`, `find_defect`) and
one rule, customers before orders — more than it needs: a profile plus three
queries against the warehouse keys also reaches 1.0 (a test checks this). Every rule
is written in the contract; nothing here needs judgement beyond applying it
literally.

The lazy version of the job is in the attacker table below as `profile-only`: it
profiles every batch and applies every rule a profile can show, but never queries
the warehouse. It scores **0.8750 while loading 10 of the 24 bad batches** (a
revenue error of $105,467.75) — re-sent orders, re-added customers and orders for
customers who do not exist only show up against the warehouse. That gap, not the
mean, is what separates an agent that checks from one that looks.

And the scale is coarse at the top: a batch that ends in the wrong place costs 0.25
to 0.4 on its case (0.008 to 0.0125 on the 32-case mean), and a wrong citation on a
correctly stopped batch costs 0.15. A model that looks before it acts and reports
truthfully has to put at least five of the 64 batches in the wrong place to fall
below 0.95. Expect capable models to crowd near 1.0. Whether the remaining spread
separates them is what a real-model run has to show.

## Real models

**Not measured yet.** Nothing in this README comes from a model run. Results will be
added here with the date they were measured. To run one yourself (needs an API key
for the provider you choose, and costs money):

```bash
pip install -e .                  # Python 3.11–3.13; or: prime env install jigonyoo/bad-batch-gate
vf-eval bad-batch-gate -m gpt-4.1-mini -n 32 -r 3
```

`load_environment` takes `max_turns` (default 30), `eval_file` and `train_file`;
pass them with `-a '{"max_turns": 40}'`.

## Adversarial check

`python3 scripts/run_attacks.py`. Several attackers are *oracles*: they are given
the truth labels, which no real agent has. If an agent that knows the answer still
cannot score by claiming it, the grader is reading the warehouse and not the story.

| Attacker | Trick | Reward |
|---|---|---:|
| careful (target) | — | 1.0000 |
| report-liar (oracle) | load everything, report the right answer | 0.3844 |
| report-only (oracle) | touch nothing, report the right answer | 0.0375 |
| load-then-quarantine (oracle) | load everything, then quarantine the bad ones with the right citation | 0.6844 |
| citation-spray | quarantine everything, trying every (column, defect type) pair | 0.5375 |
| sql-write | `DELETE` / `UPDATE` / `DROP` / stacked statements / `PRAGMA` / `ATTACH` through `run_sql` | 0.5344 |
| inspect-theatre | profile every batch, then load everything | 0.6844 |
| double-load | careful checks, but every clean batch loaded twice | 0.6875 |
| orders-first | careful checks in listed order, not customers first | 0.9750 |
| profile-only | profile every batch and apply what a profile shows; never query the warehouse | 0.8750 |

Holes found and closed while building. The scores quoted inside items 1–5 were
measured on earlier versions of this code, which no longer exist; they come from
the review logs and cannot be re-run from this repository.

1. **Quarantine after load.** `evidence` first credited any true citation, so an
   agent that loaded a bad batch and flagged it afterwards scored 0.734. A
   quarantine now earns evidence only if the batch never reached the warehouse.
2. **Load order was never tested.** Batches were shuffled, and in every case
   where new customers and their orders arrive together, the customers happened to
   be listed first — an agent that ignored load order scored a perfect 1.000. Two
   of those cases now list the orders first. A later review found the grader still
   only noticed the order in which batches were *checked*: an agent that judged
   every batch correctly but *loaded* the orders before their customers scored 1.0,
   because the revenue mart joins the same rows either way. `disposition` now also
   fails a clean orders batch that was loaded while its customers were not yet in
   the warehouse.

3. **Doing nothing paid.** An agent that touched nothing and filed an empty report
   scored 0.503 — above both baselines — because a bad batch left in staging counted
   as handled. Leaving a batch untouched now counts as not handling it (0.2375).
4. **Runs that never finished kept their credit.** A rollout cut off by a timeout
   or the turn limit scored up to 0.5 on a case. Now: no report, no credit.
5. **The answer leaked through metadata.** The defective batch always got the id
   suffix `-1`, only re-sent files carried a "retried" source note, a customers batch
   listed first was always the bad one, a duplicated row made the batch one row
   longer, and single-batch cases were mostly defective. An agent that never read a
   row scored 0.744 on those priors alone. Now every case has two batches, clean
   customers batches are as common as bad ones, duplicates replace a row instead of
   adding one, batch ids are assigned after shuffling and source notes are drawn
   independently. A third review then found one more: order ids were counted up, and
   the defective batch is built first, so it always held the smallest ids — an agent
   that quarantined the batch with the lowest id scored 0.825. Ids are now drawn at
   random. A fourth review found two more: orphan customer ids were always `C9xx` and
   re-added customers always `C0xx`, and re-sent rows were the only orders dated
   before the 18th — reading only min/max from a profile scored 0.838. Customer ids
   now come from one shared random range and all orders share one date range. A fifth
   review found the last one it could: the re-added customer carried a `+crm` email
   and an older signup date (44 of 44). It is now an exact re-send. A test checks each
   of these, at the level of the individual row.
6. **One query could stall everything.** `run_sql` runs on the event loop. A
   recursive CTE with no stop hung every rollout in the process; `zeroblob()` could
   allocate gigabytes; after that was fixed, a second review found that a single
   `LIKE` over long strings could take seconds without tripping a step counter, and
   that a 1,000-column row of 100 KB strings exhausted memory. Queries now stop after
   2M SQLite steps **or 2 seconds**, values are capped at 20 KB, `LIKE` patterns at
   200 characters, results at 64 columns and 50 rows of cells cut at 200 characters,
   and `zeroblob`/`randomblob`/`load_extension` are refused. Below Python 3.11
   SQLite's limits cannot be set from Python, so there `printf`, `format`,
   `group_concat`, `string_agg`, `replace` and `char` are refused as well — **but that
   is not equivalent protection**: a recursive query that doubles a string with `||`
   still grows without limit on 3.10, and in review such queries ran for many
   seconds, well past the 2-second budget. The verifiers path requires Python 3.11+; do not expose `run_sql`
   to untrusted agents on anything older.
7. **Naming a table counted as reading it.** `inspected` matched the staging table's
   name anywhere in the query text, so `SELECT 1 -- stg_…` counted as a look. It now
   uses the tables SQLite resolves the query against — a real reference in `FROM`,
   not a name in a comment or a string. (A query like `SELECT 1 FROM stg_… WHERE 0`
   still counts; see the limits below.)
8. **A date trap the contract never set.** The prompt used to name an ingest date
   inside September, so valid orders later that month looked like they came from
   the future. The run date is now 1 October, after the window closes.

Items 3–8 and the second half of item 2 were found by independent reviewers who
were given the code and the commands, not the author's conclusions. Running the
offline scripts on Python 3.10 found one more: the authorizer could not be switched
off there with `set_authorizer(None)`, so every load failed. It is now installed once
and switched by a flag. The last three reviews also planted bugs of their own in the
grader, the tools, the data and the wiring, and each round found some that no test
caught — among them the very rules in items 1, 3 and 7. Each of those now has a
test, checked by putting the bug back and watching the suite fail. That covers the
bugs the reviews thought of, not every bug there could be. Two tests also pin the
tables in this README to the scripts' output.

`orders-first` is not a cheat; it is a realistic mistake, and only 2 of the 32
eval cases always catch it (two more can, depending on how their batches are
shuffled; in this draw neither does), so it sits 0.025 below the target rather than
the 0.05 the other attackers clear. The same two cases are the only ones that catch
an agent that judges correctly but loads in the listed order. The test suite holds
every other attacker at least 0.05 below `careful`. `profile-only` is not a cheat
either; it is there to show what skipping the warehouse costs.

`run_sql` cannot write: a SQLite authorizer denies everything except reads. The six
attempts in the table, plus `INSERT` and `CREATE TABLE`, are each a test.

## What this does not measure

- **Real exports.** Batches are 3–20 generated rows over two tables. A profile of a
  real batch is harder to read, and real contracts are vaguer than this one.
- **Hostile SQL below Python 3.11.** See item 6: the value-length limit needs 3.11+.
- **Unit errors under the cap.** The contract's amount rule is a fixed cap (50,000),
  and every planted cents/dollars error lands above it. A ×100 error on a small
  order would stay under the cap and pass the contract, and no such case is in the
  data: in the eval split, 157 of the 562 positive clean orders are $500 or less.
- **Partial loads.** Quarantine is all-or-nothing per batch. Loading the good rows
  and holding back the bad ones — often the right call in production — is not
  modelled.
- **One defect per bad batch.** `evidence` checks a single citation. It does not
  check that the agent found *every* problem.
- **Whether the inspection was relevant.** Any `profile_batch` or any query that
  references the staging table in `FROM` counts as having looked, whatever it checks —
  even `WHERE 0`.
- **Judgement under ambiguity.** Every rule is written down. Nothing here asks an
  agent to decide whether an odd-looking value is a defect the contract forgot.
- **Model behaviour.** See *Real models* — not run yet.
- **Rollouts scored outside the rubric.** Each rollout's in-memory warehouse is
  released when it is scored. If you run rollouts with scoring turned off, those
  warehouses stay in memory until the process ends.

## Run it

From a clone of this repository. Offline checks, no key, no network — the two
scripts need nothing but the standard library:

```bash
python3 scripts/run_report.py
python3 scripts/run_attacks.py
python3 -m venv .venv-offline && . .venv-offline/bin/activate
pip install pytest
python -m pytest tests/test_dataset.py tests/test_grader.py   # 100 tests (3.10: 96 + 4 skipped)
deactivate
```

With `verifiers` (Python 3.11–3.13, which is what verifiers 0.3.1 supports):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e . pytest
python -m pytest            # 117 tests
```

The wheel on the Hub carries the package only; the scripts and tests are in this
repository and in the sdist.

Regenerate the data (both splits byte-identical to the committed files; a test
checks the bytes):

```bash
python3 bad_batch_gate/dataset.py
```

## Layout

```
bad_batch_gate/
  desk.py          staging, warehouse, the loader, the hash-chained ledger
  grader.py        four scored terms + two metrics, all from the warehouse
  dataset.py       eight families, fixed seeds
  agents.py        naive / careful / refuse-all
  attackers.py     the eight attackers and the profile-only baseline above (kept on purpose)
  environment.py   verifiers wiring
  data/            eval_curated.jsonl (32), train_procedural.jsonl (320)
scripts/           run_report.py, run_attacks.py
tests/             test_dataset.py, test_grader.py, test_environment.py
```

Built with AI assistance, and reviewed by separate AI agents that were given the
code and the commands but not the author's conclusions. Every number in the
tables and in *What a perfect score takes* comes from the commands above; the
history in items 1–5 comes from the review logs.

MIT licence.
