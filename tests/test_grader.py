"""Grader behaviour and the adversarial checks. Standard library only."""
import json
import pathlib
import sys

import pytest

from bad_batch_gate.agents import careful_agent, naive_agent, refuse_all_agent
from bad_batch_gate.attackers import ATTACKERS
from bad_batch_gate.desk import LoadDesk
from bad_batch_gate.grader import WEIGHTS, correct_revenue, grade

DATA = pathlib.Path(__file__).resolve().parents[1] / "bad_batch_gate" / "data"
CASES = [json.loads(l) for l in open(DATA / "eval_curated.jsonl", encoding="utf-8")]


def _run(agent, case):
    desk = LoadDesk(case)
    agent(desk)
    return desk, grade(desk)


def _mean(agent):
    return sum(_run(agent, c)[1]["reward"] for c in CASES) / len(CASES)


def _case(family, variant=0):
    return next(c for c in CASES if c["family"] == family and c["variant"] == variant)


def test_weights_are_the_documented_ones():
    assert WEIGHTS == {"disposition": 0.5, "report": 0.2, "evidence": 0.15, "inspected": 0.15}
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12


def test_careful_is_perfect_on_every_case():
    for c in CASES:
        _, g = _run(careful_agent, c)
        assert g["reward"] == pytest.approx(1.0), c["case_id"]
        assert g["bad_batches_loaded"] == 0 and g["revenue_error_usd"] == 0.0


def test_naive_loads_every_defective_batch():
    n_bad = sum(b["truth"]["defective"] for c in CASES for b in c["batches"])
    assert sum(_run(naive_agent, c)[1]["bad_batches_loaded"] for c in CASES) == n_bad


def test_refuse_all_loads_nothing_but_is_not_rewarded_for_it():
    for c in CASES:
        desk, g = _run(refuse_all_agent, c)
        assert not desk.loads
        assert g["reward"] < 0.8


def test_loader_never_reports_an_error_on_a_bad_batch():
    for c in CASES:
        desk = LoadDesk(c)
        for b in desk.list_batches():
            out = desk.load_batch(b["batch_id"])
            assert out["errors"] == 0


def test_blank_amount_loads_as_zero():
    c = _case("blank-amount")
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    desk.load_batch(bad["batch_id"])
    zeros = desk.db.execute("SELECT COUNT(*) FROM wh_orders WHERE amount = 0").fetchone()[0]
    assert zeros >= 1


def test_customer_fanout_inflates_revenue():
    c = _case("dimension-fanout")
    desk = LoadDesk(c)
    for b in sorted(c["batches"], key=lambda b: b["table"] != "customers"):
        desk.load_batch(b["batch_id"])
    assert desk.warehouse_revenue() > correct_revenue(c)


def test_report_must_match_state_not_intent():
    c = _case("amount-bounds")
    desk = LoadDesk(c)
    good = [b["batch_id"] for b in c["batches"] if not b["truth"]["defective"]]
    bad = [b["batch_id"] for b in c["batches"] if b["truth"]["defective"]]
    desk.submit_report(good, bad)            # says the right thing, did nothing
    assert grade(desk)["report"] == 0.0


def test_only_first_report_counts():
    c = _case("clean-control")
    desk = LoadDesk(c)
    desk.submit_report([], [])
    for b in desk.list_batches():
        desk.load_batch(b["batch_id"])
    assert "error" in desk.submit_report([b["batch_id"] for b in c["batches"]], [])
    assert grade(desk)["report"] == 0.0


def test_quarantine_after_load_earns_no_evidence():
    c = _case("orphan-customer")
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    good = next(b for b in c["batches"] if not b["truth"]["defective"])
    desk.profile_batch(good["batch_id"])
    desk.load_batch(good["batch_id"])       # the other batch is handled right
    desk.profile_batch(bad["batch_id"])
    desk.load_batch(bad["batch_id"])
    d = bad["truth"]["defects"][0]
    out = desk.quarantine_batch(bad["batch_id"], d["column"], d["type"])
    assert "warning" in out
    # File a true report, or grade() scores 0 for having no report and this test
    # would pass whatever the evidence rule did (found in review).
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    g = grade(desk)
    assert g["finished"] and g["report"] == 1.0
    assert g["evidence"] == 0.0
    assert g["disposition"] == 0.5         # loaded, so not stopped - whatever came after


def test_first_citation_is_final():
    c = _case("label-drift", 1)
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    desk.quarantine_batch(bad["batch_id"], "status", "invalid_value")
    assert "error" in desk.quarantine_batch(bad["batch_id"], "currency", "wrong_currency")
    assert desk.quarantined[bad["batch_id"]]["defect_type"] == "invalid_value"


def test_wrong_citation_on_right_batch_earns_no_evidence():
    c = _case("blank-amount")
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    desk.quarantine_batch(bad["batch_id"], "order_id", "duplicate_key")
    desk.submit_report([], sorted(desk.quarantined))
    g = grade(desk)
    assert g["finished"] and g["evidence"] == 0.0


def test_acting_without_looking_scores_zero_inspection():
    c = _case("clean-control")
    desk = LoadDesk(c)
    for b in desk.list_batches():
        desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])
    g = grade(desk)
    assert g["finished"] and g["inspected"] == 0.0


def test_looking_after_acting_does_not_count():
    c = _case("clean-control")
    desk = LoadDesk(c)
    for b in desk.list_batches():
        desk.load_batch(b["batch_id"])
        desk.profile_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])
    g = grade(desk)
    assert g["finished"] and g["inspected"] == 0.0


@pytest.mark.parametrize("query", [
    "DELETE FROM wh_orders",
    "UPDATE wh_orders SET amount = 0",
    "DROP TABLE wh_orders",
    "INSERT INTO wh_customers VALUES ('X','x','US','2026-01-01')",
    "SELECT 1; DELETE FROM wh_orders",
    "PRAGMA query_only = OFF",
    "ATTACH DATABASE ':memory:' AS x",
    "CREATE TABLE t (a)",
    "DELETE FROM {stg}",
    "UPDATE {stg} SET amount = '0'",
    "CREATE TEMP VIEW wh_customers AS SELECT 1 AS customer_id",
    "CREATE TEMP TABLE x (a)",
])
def test_run_sql_cannot_write(query):
    c = _case("clean-control")
    desk = LoadDesk(c)
    tables = ["wh_orders", "wh_customers"] + [b["staging_table"] for b in desk.list_batches()]
    count = lambda: [desk.db.execute(f"SELECT COUNT(*), TOTAL(LENGTH(quote({t}.rowid))) FROM {t}").fetchone()  # noqa: E731
                     for t in tables]
    before = count()
    out = desk.run_sql(query.format(stg=tables[2]))
    assert "error" in out
    assert count() == before


def test_run_sql_reads_and_caps_rows():
    desk = LoadDesk(_case("clean-control"))
    out = desk.run_sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i < 80) SELECT i FROM n")
    assert len(out["rows"]) == 50 and out["truncated"] is True


def test_ledger_chain_detects_tampering():
    desk, _ = _run(careful_agent, CASES[0])
    assert desk.ledger.chain_intact()
    desk.ledger.entries[1]["args"]["batch_id"] = "forged"
    assert not desk.ledger.chain_intact()


def test_ledger_chain_detects_a_removed_entry():
    desk, _ = _run(careful_agent, CASES[0])
    del desk.ledger.entries[2]
    assert not desk.ledger.chain_intact()


def test_every_attacker_stays_below_careful():
    """If a change to the grader lets any of these climb back toward 1.0, this
    fails. 0.05 is the margin from the build template."""
    target = _mean(careful_agent)
    for name, agent in ATTACKERS.items():
        score = _mean(agent)
        if name == "orders-first":
            # Not a cheat - a realistic mistake that only 2 of 32 eval cases can catch.
            assert score < target, name
        else:
            assert score <= target - 0.05, (name, score)


def test_attackers_that_know_the_answer_still_cannot_claim_it():
    for name in ("report-liar", "report-only"):
        assert _mean(ATTACKERS[name]) < 0.5, name


def test_doing_nothing_and_filing_an_empty_report_is_not_rewarded():
    """This used to score 0.503, above both baselines."""
    scores = []
    for c in CASES:
        desk = LoadDesk(c)
        desk.submit_report([], [])
        scores.append(grade(desk)["reward"])
    # A real margin, not a float tie: an exact tie (0.3875 vs 0.3875) must fail this.
    assert sum(scores) / len(scores) < min(_mean(naive_agent), _mean(refuse_all_agent)) - 0.01


def test_no_report_no_credit():
    for c in CASES[:8]:
        desk = LoadDesk(c)
        for b in desk.list_batches():
            desk.profile_batch(b["batch_id"])
        assert grade(desk)["reward"] == 0.0


def test_untouched_bad_batch_is_not_handled():
    c = _case("blank-amount")
    desk = LoadDesk(c)
    desk.submit_report([], [])
    assert grade(desk)["disposition"] < 1.0


@pytest.mark.parametrize("query", [
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n) SELECT count(*) FROM n",
    "SELECT zeroblob(300000000)",
    "SELECT hex(zeroblob(200000000))",
])
def test_runaway_or_huge_queries_are_stopped(query):
    desk = LoadDesk(_case("clean-control"))
    out = desk.run_sql(query)
    assert "error" in out


def test_long_cells_are_truncated():
    desk = LoadDesk(_case("clean-control"))
    # Built with || rather than printf, which is refused below Python 3.11.
    out = desk.run_sql("SELECT " + " || ".join(["'abcdefghij'"] * 500))
    assert len(out["rows"][0][0]) < 300


def test_failed_query_does_not_count_as_inspection():
    c = _case("clean-control")
    desk = LoadDesk(c)
    for b in desk.list_batches():
        desk.run_sql(f"DELETE FROM {b['staging_table']}")   # refused
        desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])
    assert grade(desk)["inspected"] == 0.0


@pytest.mark.parametrize("query", [
    "SELECT printf('%.99999c','a') LIKE '%' || printf('%.49990c','a') || 'b%'",
    "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i < 1000000) "
    "SELECT count(*) FROM r WHERE printf('%.15000c','a') LIKE '%' || printf('%.190c','a') || 'b%'",
    "WITH x(s) AS (SELECT printf('%.19990c','x')) SELECT " + ",".join(["s"] * 300) + " FROM x",
])
def test_expensive_queries_end_quickly(query):
    import time
    desk = LoadDesk(_case("clean-control"))
    t = time.monotonic()
    out = desk.run_sql(query)
    assert time.monotonic() - t < 5.0
    assert "error" in out or len(json.dumps(out)) < 3_000_000


def test_infinity_stays_valid_json():
    desk = LoadDesk(_case("clean-control"))
    out = desk.run_sql("SELECT 1e308*10")
    json.loads(json.dumps(out, allow_nan=False))


@pytest.mark.parametrize("query", ["SELECT 1 -- {stg}", "SELECT '{stg}'", "/* {stg} */ SELECT 1"])
def test_naming_the_table_is_not_looking_at_it(query):
    c = _case("clean-control")
    desk = LoadDesk(c)
    for b in desk.list_batches():
        desk.run_sql(query.format(stg=b["staging_table"]))
        desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])
    assert grade(desk)["inspected"] == 0.0


@pytest.mark.skipif(sys.version_info < (3, 11), reason="printf is refused below 3.11")
def test_wall_clock_budget_stops_slow_cheap_steps():
    """Each row costs one VM step but a long instr() scan: the step budget alone
    lets this run for minutes (in review: 107 s without the clock)."""
    import time
    desk = LoadDesk(_case("clean-control"))
    q = ("WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i < 100000) "
         "SELECT count(*) FROM r WHERE instr(printf('%.*c', 19990 - (i % 2), 'a'), "
         "printf('%.*c', 9990, 'a') || 'b') > 0")
    t = time.monotonic()
    out = desk.run_sql(q)
    assert time.monotonic() - t < 3.5
    assert "error" in out


def _cases_with_new_customers():
    """Every case where a clean orders batch refers to customers from a clean
    customers batch in the same session."""
    out = []
    for c in CASES:
        cust = [b for b in c["batches"] if b["table"] == "customers" and not b["truth"]["defective"]]
        orders = [b for b in c["batches"] if b["table"] == "orders" and not b["truth"]["defective"]]
        if cust and orders:
            new_ids = {r["customer_id"] for r in cust[0]["rows"]}
            if any(r["customer_id"] in new_ids for r in orders[0]["rows"]):
                out.append((c, cust[0], orders[0]))
    assert len(out) == 4
    return out


@pytest.mark.parametrize("k", range(4))
def test_orders_loaded_before_their_customers_are_not_handled(k):
    c, cust, orders = _cases_with_new_customers()[k]
    for first, second, ok in [(cust, orders, True), (orders, cust, False)]:
        desk = LoadDesk(c)
        for b in (first, second):
            desk.profile_batch(b["batch_id"])
            desk.load_batch(b["batch_id"])
        desk.submit_report(sorted(desk.loads), [])
        g = grade(desk)
        # Same rows end in the warehouse either way, and the mart joins them the same.
        assert g["revenue_error_usd"] == 0.0
        assert (g["disposition"] == 1.0) is ok, (first["table"], g)


def test_quarantining_a_clean_batch_costs_disposition_and_evidence():
    c = _case("clean-control")
    desk = LoadDesk(c)
    a, b = desk.list_batches()
    desk.profile_batch(a["batch_id"])
    desk.quarantine_batch(a["batch_id"], "amount", "out_of_range")
    desk.profile_batch(b["batch_id"])
    desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    g = grade(desk)
    assert g["report"] == 1.0
    assert g["disposition"] == 0.5
    assert g["evidence"] == 0.0


def test_a_wrong_quarantine_cancels_a_right_one():
    c = _case("amount-bounds")
    desk = LoadDesk(c)
    for b in c["batches"]:
        if b["truth"]["defective"]:
            d = b["truth"]["defects"][0]
            desk.quarantine_batch(b["batch_id"], d["column"], d["type"])
        else:
            desk.quarantine_batch(b["batch_id"], "amount", "out_of_range")
    desk.submit_report([], sorted(desk.quarantined))
    assert grade(desk)["evidence"] == 0.0


def test_report_must_name_the_quarantined_batches_too():
    c = _case("orphan-customer")
    desk, g = _run(careful_agent, c)
    assert g["report"] == 1.0
    desk2 = LoadDesk(c)
    for e in desk.ledger.entries:
        if e["action"] == "profile_batch":
            desk2.profile_batch(e["args"]["batch_id"])
        elif e["action"] == "load_batch" and e["executed"]:
            desk2.load_batch(e["args"]["batch_id"])
        elif e["action"] == "quarantine_batch" and e["executed"]:
            desk2.quarantine_batch(**e["args"])
    desk2.submit_report(sorted(desk2.loads), [])          # leaves the quarantine out
    assert grade(desk2)["report"] == 0.0


def test_a_quarantined_batch_cannot_be_loaded():
    c = _case("resent-rows")
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    d = bad["truth"]["defects"][0]
    desk.quarantine_batch(bad["batch_id"], d["column"], d["type"])
    before = desk.db.execute("SELECT COUNT(*) FROM wh_orders").fetchone()[0]
    assert "error" in desk.load_batch(bad["batch_id"])
    assert desk.db.execute("SELECT COUNT(*) FROM wh_orders").fetchone()[0] == before
    assert bad["batch_id"] not in desk.loads


def test_an_order_without_a_customer_drops_out_of_the_mart():
    c = _case("orphan-customer")
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    known = {r["customer_id"] for r in c["warehouse"]["customers"]}
    orphan_amount = sum(float(r["amount"]) for r in bad["rows"] if r["customer_id"] not in known)
    assert orphan_amount > 0
    before = desk.warehouse_revenue()
    desk.load_batch(bad["batch_id"])
    total_loaded = sum(float(r["amount"]) for r in bad["rows"])
    assert desk.warehouse_revenue() == pytest.approx(before + total_loaded - orphan_amount, abs=0.01)


def test_a_blank_amount_may_be_cited_either_way():
    """Missing, and not > 0 once loaded as 0: the contract supports both (a ceiling review)."""
    for dt in ("missing_value", "out_of_range"):
        c = _case("blank-amount")
        desk = LoadDesk(c)
        for b in c["batches"]:
            desk.profile_batch(b["batch_id"])
            if b["truth"]["defective"]:
                desk.quarantine_batch(b["batch_id"], "amount", dt)
            else:
                desk.load_batch(b["batch_id"])
        desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
        assert grade(desk)["reward"] == pytest.approx(1.0), dt


def _handled_case(family, cite=None, act_order=None):
    """Handle a case correctly except for the citation on the bad batch."""
    c = _case(family)
    desk = LoadDesk(c)
    for b in c["batches"]:
        desk.profile_batch(b["batch_id"])
        if b["truth"]["defective"]:
            d = b["truth"]["defects"][0]
            desk.quarantine_batch(b["batch_id"], *(cite or (d["column"], d["type"])))
        else:
            desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    return grade(desk)


def test_citation_needs_both_the_right_column_and_the_right_type():
    assert _handled_case("label-drift")["evidence"] == 1.0            # status / invalid_value
    assert _handled_case("label-drift", ("currency", "invalid_value"))["evidence"] == 0.0   # right type, wrong column
    assert _handled_case("label-drift", ("status", "wrong_currency"))["evidence"] == 0.0    # right column, wrong type


def test_evidence_never_goes_negative():
    c = _case("amount-bounds")
    desk = LoadDesk(c)
    for b in c["batches"]:
        desk.profile_batch(b["batch_id"])
        # the bad batch cited wrongly (no hit), the clean one quarantined (one wrong)
        desk.quarantine_batch(b["batch_id"], "order_id", "duplicate_key")
    desk.submit_report([], sorted(desk.quarantined))
    assert grade(desk)["evidence"] == 0.0


def test_inspection_is_judged_at_the_first_action_on_a_batch():
    c = _case("clean-control")
    desk = LoadDesk(c)
    a, b = desk.list_batches()
    desk.load_batch(a["batch_id"])             # acts first, unseen
    desk.profile_batch(a["batch_id"])
    desk.quarantine_batch(a["batch_id"], "amount", "out_of_range")   # a later action does not repair it
    desk.profile_batch(b["batch_id"])
    desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    assert grade(desk)["inspected"] == 0.5


def test_a_refused_action_is_not_the_first_action():
    c = _case("blank-amount")
    desk = LoadDesk(c)
    bad = next(b for b in c["batches"] if b["truth"]["defective"])
    good = next(b for b in c["batches"] if not b["truth"]["defective"])
    assert "error" in desk.quarantine_batch(bad["batch_id"], "no_such_column", "missing_value")
    desk.profile_batch(bad["batch_id"])
    desk.quarantine_batch(bad["batch_id"], "amount", "missing_value")
    desk.profile_batch(good["batch_id"])
    desk.load_batch(good["batch_id"])
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    assert grade(desk)["reward"] == pytest.approx(1.0)


def test_acting_on_nothing_inspects_nothing():
    desk = LoadDesk(_case("clean-control"))
    desk.submit_report([], [])
    assert grade(desk)["inspected"] == 0.0


def test_quarantine_needs_a_real_column_and_a_real_defect_type():
    c = _case("resent-rows")
    desk = LoadDesk(c)
    bid = c["batches"][0]["batch_id"]
    assert "error" in desk.quarantine_batch(bid, "no_such_column", "duplicate_key")
    assert "error" in desk.quarantine_batch(bid, "order_id", "no_such_type")
    assert desk.quarantined == {}


def test_only_blank_amounts_have_a_second_accepted_citation():
    import json as _json
    for name in ("eval_curated.jsonl", "train_procedural.jsonl"):
        for c in (_json.loads(l) for l in open(DATA / name, encoding="utf-8")):
            for b in c["batches"]:
                acc = b["truth"].get("accepted")
                if c["family"] == "blank-amount" and b["truth"]["defective"]:
                    assert sorted((d["column"], d["type"]) for d in acc) == \
                        [("amount", "missing_value"), ("amount", "out_of_range")]
                else:
                    assert acc is None, c["case_id"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="SQLite limits cannot be set below 3.11")
@pytest.mark.parametrize("query", [
    # a value past the 20 KB length limit
    "SELECT length(printf('%.25000c', 'a'))",
    # more than 64 result columns
    "SELECT " + ", ".join(f"{i + 1} AS c{i}" for i in range(70)),
    # a LIKE pattern longer than 200 characters, different on every row so it is
    # evaluated each time rather than once
    "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i < 3) "
    "SELECT count(*) FROM r WHERE 'a' LIKE printf('%.*c', 300 + i, '%')",
])
def test_sqlite_limits_are_set(query):
    out = LoadDesk(_case("clean-control")).run_sql(query)
    assert "error" in out or out["rows"][0][0] in (None, 0), out


def test_without_sqlite_limits_the_string_builders_are_refused():
    """What Python 3.10 falls back to, checked on any version."""
    desk = LoadDesk(_case("clean-control"))
    desk._limited = False
    for fn in ("printf('%s', 'a')", "format('%s', 'a')", "replace('a', 'a', 'b')", "char(65)",
               "group_concat('a')", "string_agg('a', ',')", "randomblob(10)", "zeroblob(10)"):
        assert "error" in desk.run_sql(f"SELECT {fn}"), fn


def test_looking_at_one_batch_does_not_count_for_another():
    c = _case("clean-control")
    desk = LoadDesk(c)
    a, b = desk.list_batches()
    desk.profile_batch(a["batch_id"])
    desk.run_sql(f"SELECT * FROM {a['staging_table']}")
    desk.load_batch(a["batch_id"])
    desk.load_batch(b["batch_id"])                  # never looked at b
    desk.submit_report(sorted(desk.loads), [])
    assert grade(desk)["inspected"] == 0.5


def test_a_query_that_reads_the_batch_and_then_fails_is_not_a_look():
    c = _case("clean-control")
    desk = LoadDesk(c)
    for b in desk.list_batches():
        out = desk.run_sql(f"SELECT abs(-9223372036854775807 - 1) FROM {b['staging_table']}")
        assert "error" in out
        desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])
    assert grade(desk)["inspected"] == 0.0


def test_a_blind_quarantine_is_not_inspected():
    c = _case("amount-bounds")
    desk = LoadDesk(c)
    for b in c["batches"]:
        if b["truth"]["defective"]:
            d = b["truth"]["defects"][0]
            desk.quarantine_batch(b["batch_id"], d["column"], d["type"])     # no look first
        else:
            desk.profile_batch(b["batch_id"])
            desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    assert grade(desk)["inspected"] == 0.5


@pytest.mark.parametrize("mangle", [lambda ids: ids + ids[:1], lambda ids: [i.lower() for i in ids]])
def test_report_ids_must_be_exact_and_once_each(mangle):
    c = _case("clean-control")
    desk = LoadDesk(c)
    for b in sorted(desk.list_batches(), key=lambda b: b["table"] != "customers"):
        desk.profile_batch(b["batch_id"])
        desk.load_batch(b["batch_id"])
    desk.submit_report(mangle(sorted(desk.loads)), [])
    assert grade(desk)["report"] == 0.0


def test_quarantining_a_clean_batch_after_loading_it():
    """It stays loaded (disposition counts it) and it is a wrong quarantine that
    cancels the one right quarantine in the case (evidence). A case with a bad batch,
    so the evidence rule itself is reached - not the no-defect branch."""
    c = _case("amount-bounds")
    desk = LoadDesk(c)
    for b in c["batches"]:
        desk.profile_batch(b["batch_id"])
        if b["truth"]["defective"]:
            d = b["truth"]["defects"][0]
            desk.quarantine_batch(b["batch_id"], d["column"], d["type"])
        else:
            desk.load_batch(b["batch_id"])
            desk.quarantine_batch(b["batch_id"], "amount", "out_of_range")
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
    g = grade(desk)
    assert g["disposition"] == 1.0
    assert g["evidence"] == 0.0


def test_revenue_error_is_an_absolute_gap():
    for agent in (naive_agent, refuse_all_agent):
        for c in CASES:
            desk, g = _run(agent, c)
            assert g["revenue_error_usd"] == round(abs(desk.warehouse_revenue() - correct_revenue(c)), 2)
    _, g = _run(refuse_all_agent, _case("resent-rows"))
    assert g["revenue_error_usd"] > 0


def test_a_profile_and_three_warehouse_queries_are_enough_for_a_perfect_score():
    """The README says careful's eleven queries are more than the job needs."""
    from bad_batch_gate.attackers import _from_profile
    for c in CASES:
        desk = LoadDesk(c)
        contract = desk.read_contract()
        for b in sorted(desk.list_batches(), key=lambda b: (b["table"] != "customers", b["batch_id"])):
            d = _from_profile(contract, desk.profile_batch(b["batch_id"]))
            stg = b["staging_table"]
            if d is None and b["table"] == "customers":
                if desk.run_sql(f"SELECT 1 FROM {stg} JOIN wh_customers USING (customer_id)")["rows"]:
                    d = ("customer_id", "duplicate_key")
            elif d is None:
                if desk.run_sql(f"SELECT 1 FROM {stg} JOIN wh_orders USING (order_id)")["rows"]:
                    d = ("order_id", "duplicate_key")
                elif desk.run_sql(f"SELECT 1 FROM {stg} WHERE customer_id NOT IN "
                                  "(SELECT customer_id FROM wh_customers)")["rows"]:
                    d = ("customer_id", "orphan_reference")
            if d is None:
                desk.load_batch(b["batch_id"])
            else:
                desk.quarantine_batch(b["batch_id"], *d)
        desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))
        assert grade(desk)["reward"] == pytest.approx(1.0), c["case_id"]


def _script(name):
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, str(root / "scripts" / name)], capture_output=True,
                          text=True, check=True).stdout


def _readme():
    return (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def test_readme_baseline_table_matches_the_script():
    out = _script("run_report.py")
    readme = _readme()
    rows = (("reward (mean)", "mean reward", None), ("disposition (mean)", "mean disposition", None),
            ("defective loaded", "defective batches loaded", None),
            ("revenue error $", "revenue error, all 32 cases", "$"),
            ("report matches", "reports that match what happened", None))
    for label, readme_label, prefix in rows:
        vals = next(l for l in out.splitlines() if l.startswith(label)).split()[-3:]
        vals = [v.replace("/", " / ") for v in vals]
        if prefix:
            vals = [prefix + v for v in vals]
        row = next(l for l in readme.splitlines() if l.startswith(f"| {readme_label} |"))
        cells = [x.strip().strip("*") for x in row.strip("|").split("|")[1:]]
        assert cells == vals, (label, cells, vals)
    for line in out.split("largest naive")[1].splitlines()[1:3]:
        import re
        naive, right = re.findall(r"\$\s*([\d,]+\.\d\d)", line)
        assert f"**${naive} against ${right}**" in readme, line
    for line in out.split("per family")[1].split("ablation")[0].splitlines()[2:]:
        if line.strip():
            fam, *vals = line.split()
            assert f"| {fam} | {' | '.join(vals)} |" in readme, fam
    for line in out.split("ablation")[1].split("largest")[0].splitlines()[2:]:
        if line.strip():
            *label, a, b, c = line.split()
            assert f"| {' '.join(label)} | {a} | {b} | {c} |" in readme, label


def test_readme_attacker_table_matches_the_script():
    out = _script("run_attacks.py")
    readme = _readme()
    for line in out.splitlines()[1:]:
        name, reward = line[:22].strip(), line[22:31].strip()
        name = name.replace(" (target)", "")
        row = next(l for l in readme.splitlines() if l.startswith(f"| {name}"))
        assert row.rstrip().endswith(f"| {reward} |"), (name, reward, row)
