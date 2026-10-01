"""Dataset invariants. Standard library only."""
import json
import pathlib
from collections import Counter

from bad_batch_gate.dataset import CONTRACT, CONTROL_FAMILIES, FAMILIES, build_eval, build_train, write
from bad_batch_gate.agents import find_defect
from bad_batch_gate.desk import COLUMNS, DEFECT_TYPES, LoadDesk

DATA = pathlib.Path(__file__).resolve().parents[1] / "bad_batch_gate" / "data"


def _load(name):
    return [json.loads(l) for l in open(DATA / name, encoding="utf-8")]


def test_eval_file_matches_generator():
    """The committed eval split is exactly what the generator produces."""
    on_disk = _load("eval_curated.jsonl")
    assert [json.dumps(c, sort_keys=True) for c in on_disk] == \
           [json.dumps(c, sort_keys=True) for c in build_eval()]


def test_data_files_are_byte_identical_to_the_generator(tmp_path):
    """Not just the same JSON: the same bytes, through the same write() the README
    tells you to run, so the published sha256 holds."""
    write(tmp_path)
    for name in ("eval_curated.jsonl", "train_procedural.jsonl"):
        assert (DATA / name).read_bytes() == (tmp_path / name).read_bytes(), name


def test_train_file_matches_generator():
    on_disk = _load("train_procedural.jsonl")
    assert len(on_disk) == 320
    assert [json.dumps(c, sort_keys=True) for c in on_disk] == \
           [json.dumps(c, sort_keys=True) for c in build_train()]


def test_eval_has_four_cases_per_family():
    counts = Counter(c["family"] for c in _load("eval_curated.jsonl"))
    assert counts == {f: 4 for f in FAMILIES}


def test_case_ids_unique():
    ids = [c["case_id"] for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl")]
    assert len(ids) == len(set(ids))


def test_controls_have_no_defect_and_others_exactly_one_bad_batch():
    for c in _load("eval_curated.jsonl"):
        bad = [b for b in c["batches"] if b["truth"]["defective"]]
        if c["family"] in CONTROL_FAMILIES:
            assert bad == [], c["case_id"]
        else:
            assert len(bad) == 1, c["case_id"]
            assert len(bad[0]["truth"]["defects"]) == 1


def test_defect_labels_are_well_formed():
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        for b in c["batches"]:
            for d in b["truth"]["defects"]:
                assert d["type"] in DEFECT_TYPES
                assert d["column"] in COLUMNS[b["table"]]


def test_batch_ids_unique_within_case_and_order_ids_new_in_clean_batches():
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        ids = [b["batch_id"] for b in c["batches"]]
        assert len(ids) == len(set(ids))
        wh = {o["order_id"] for o in c["warehouse"]["orders"]}
        for b in c["batches"]:
            if b["table"] == "orders" and not b["truth"]["defective"]:
                oid = [r["order_id"] for r in b["rows"]]
                assert len(oid) == len(set(oid)) and not (set(oid) & wh), c["case_id"]


def test_contract_checks_agree_with_truth_labels_everywhere():
    """The label on every batch in both splits matches what the contract, applied
    as SQL in the right order, finds. A mislabelled batch would make the grader
    punish a correct agent."""
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        desk = LoadDesk(c)
        listed = desk.list_batches()
        for b in sorted(listed, key=lambda b: (b["table"] != "customers", b["batch_id"])):
            truth = desk.batches[b["batch_id"]]["truth"]
            found = find_defect(desk, b)
            if truth["defective"]:
                assert found is not None, (c["case_id"], b["batch_id"])
                assert {"column": found[0], "type": found[1]} in truth["defects"], (c["case_id"], found)
            else:
                assert found is None, (c["case_id"], b["batch_id"], found)
                desk.load_batch(b["batch_id"])


def test_batch_id_and_source_carry_no_signal():
    """Neither the id suffix nor the source note may predict the defective batch ."""
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    bad_suffix = Counter(b["batch_id"].rsplit("-", 1)[1] for c in cases for b in c["batches"]
                         if b["truth"]["defective"] and len(c["batches"]) > 1)
    assert len(bad_suffix) > 1, bad_suffix
    by_source = Counter((b["source"], b["truth"]["defective"]) for c in cases for b in c["batches"]
                        if b["table"] == "orders")
    for src in ("shop-api export", "pos export", "shop-api export (retry)"):
        assert by_source[(src, True)] > 0 and by_source[(src, False)] > 0, by_source


def test_no_valid_order_is_dated_after_the_run_date():
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        assert "run on 2026-10-01" in c["prompt"]
        for b in c["batches"]:
            if b["table"] == "orders" and not b["truth"]["defective"]:
                assert all(r["order_date"] <= "2026-09-30" for r in b["rows"])


def test_structure_does_not_predict_the_answer():
    """Batch count, table, listing position and row count must not give the
    defective batch away."""
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    assert {len(c["batches"]) for c in cases} == {2}
    first_customers = [c["batches"][0]["truth"]["defective"] for c in cases
                       if c["batches"][0]["table"] == "customers"]
    assert 0.25 < sum(first_customers) / len(first_customers) < 0.75
    cust = [b for c in cases for b in c["batches"] if b["table"] == "customers"]
    assert 0.25 < sum(b["truth"]["defective"] for b in cust) / len(cust) < 0.75
    for table in ("orders", "customers"):
        bad = {len(b["rows"]) for c in cases for b in c["batches"] if b["table"] == table and b["truth"]["defective"]}
        good = {len(b["rows"]) for c in cases for b in c["batches"] if b["table"] == table and not b["truth"]["defective"]}
        assert bad <= good, (table, bad - good)


def test_order_ids_do_not_rank_the_defective_batch():
    """Third review: with counted ids the defective batch always had the smallest."""
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    lower, total = 0, 0
    for c in cases:
        orders = [b for b in c["batches"] if b["table"] == "orders"]
        bad = [b for b in orders if b["truth"]["defective"]]
        good = [b for b in orders if not b["truth"]["defective"]]
        if len(bad) == 1 and len(good) == 1:
            total += 1
            lower += min(r["order_id"] for r in bad[0]["rows"]) < min(r["order_id"] for r in good[0]["rows"])
    assert total > 50 and 0.3 < lower / total < 0.7, (lower, total)


def test_ids_and_dates_do_not_mark_where_a_row_came_from():
    """Fourth review: orphan ids were always C9xx, re-added customers always C0xx,
    and re-sent rows were the only ones dated before the 18th."""
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    wh_ids = {c2["customer_id"] for c in cases for c2 in c["warehouse"]["customers"]}
    all_new = {r["customer_id"] for c in cases for b in c["batches"] if b["table"] == "customers" for r in b["rows"]}
    nums = lambda ids: {int(i[1:]) for i in ids}
    assert min(nums(wh_ids)) < 300 and max(nums(wh_ids)) > 700
    assert min(nums(all_new)) < 300 and max(nums(all_new)) > 700
    early_clean = sum(1 for c in cases for b in c["batches"]
                      if b["table"] == "orders" and not b["truth"]["defective"]
                      and min(r["order_date"] for r in b["rows"]) < "2026-09-18")
    assert early_clean > 100


def test_re_added_customer_looks_like_any_other_customer_row():
    """Fifth review: the duplicate used to carry a '+crm' email and an older date."""
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    for c in cases:
        for b in c["batches"]:
            if b["table"] == "customers":
                for r in b["rows"]:
                    assert r["email"] == f"{r['customer_id'].lower()}@example.com"
    dates_bad = [r["signup_date"] for c in cases for b in c["batches"] if b["table"] == "customers"
                 and b["truth"]["defective"] for r in b["rows"]]
    dates_good = [r["signup_date"] for c in cases for b in c["batches"] if b["table"] == "customers"
                  and not b["truth"]["defective"] for r in b["rows"]]
    assert min(dates_good) < "2026-04-01" and max(dates_bad) > "2026-08-01"


def test_orphan_ids_come_from_the_same_range_as_real_ones():
    """Row level (found in review): the range test above only looked at customers batches,
    so orphan ids drifting back to C9xx would have passed it."""
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    orphans = []
    for c in cases:
        for b in c["batches"]:
            if b["truth"]["defective"] and b["truth"]["defects"][0]["type"] == "orphan_reference":
                known = {r["customer_id"] for r in c["warehouse"]["customers"]}
                orphans += [int(r["customer_id"][1:]) for r in b["rows"] if r["customer_id"] not in known]
    assert len(orphans) > 20
    assert min(orphans) < 300 and max(orphans) > 700


def test_the_re_added_customer_is_an_exact_copy_of_the_warehouse_row():
    """Row level (found in review): an older signup date on that one row gave it away for every such row in both splits."""
    n = 0
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        wh = {r["customer_id"]: r for r in c["warehouse"]["customers"]}
        for b in c["batches"]:
            if b["table"] == "customers" and b["truth"]["defective"]:
                dup = [r for r in b["rows"] if r["customer_id"] in wh]
                assert len(dup) == 1, c["case_id"]
                assert {k: dup[0][k] for k in wh[dup[0]["customer_id"]]} == wh[dup[0]["customer_id"]], c["case_id"]
                n += 1
    assert n > 20


def test_accepted_citations_include_the_planted_ones():
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        for b in c["batches"]:
            t = b["truth"]
            if "accepted" in t:
                assert all(d in t["accepted"] for d in t["defects"])
                assert all(d["column"] in COLUMNS[b["table"]] and d["type"] in DEFECT_TYPES for d in t["accepted"])


def test_row_count_and_source_note_do_not_shift_the_odds():
    """Not just "every size appears on both sides": no row count and no source note
    may make a batch much more likely to be the bad one."""
    from collections import defaultdict
    cases = _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
    for key in (lambda b: len(b["rows"]), lambda b: b["source"]):
        groups = defaultdict(list)
        for c in cases:
            for b in c["batches"]:
                if b["table"] == "orders":
                    groups[key(b)].append(b["truth"]["defective"])
        for k, flags in groups.items():
            if len(flags) >= 50:
                assert 0.15 < sum(flags) / len(flags) < 0.6, (k, sum(flags), len(flags))


def test_a_re_sent_file_does_not_repeat_inside_itself():
    """The README says a re-sent file only clashes with the warehouse."""
    n = 0
    for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl"):
        if c["family"] == "resent-rows" and c["variant"] in (0, 1):
            bad = next(b for b in c["batches"] if b["truth"]["defective"])
            ids = [r["order_id"] for r in bad["rows"]]
            assert len(ids) == len(set(ids)), c["case_id"]
            wh = {o["order_id"] for o in c["warehouse"]["orders"]}
            assert set(ids) & wh, c["case_id"]
            n += 1
    assert n > 10


def test_legit_edge_includes_an_amount_exactly_at_the_cap():
    cap = f"{CONTRACT['limits']['max_amount']:.2f}"
    hits = [c["case_id"] for c in _load("eval_curated.jsonl") + _load("train_procedural.jsonl")
            if c["family"] == "legit-edge"
            for b in c["batches"] if b["table"] == "orders" for r in b["rows"] if r["amount"] == cap]
    assert len(hits) >= 5


def test_nothing_the_agent_can_see_carries_the_answer():
    """Truth labels, family and case id live in the case file for the grader. The
    prompt, the contract and every tool's output must not contain them."""
    for c in _load("eval_curated.jsonl"):
        desk = LoadDesk(c)
        seen = [c["prompt"], json.dumps(desk.read_contract()), json.dumps(desk.list_batches())]
        for b in desk.list_batches():
            seen.append(json.dumps(desk.profile_batch(b["batch_id"])))
            seen.append(json.dumps(desk.run_sql(f"SELECT * FROM {b['staging_table']}")))
        seen.append(json.dumps(desk.run_sql("SELECT name, sql FROM sqlite_master")))
        blob = "\n".join(seen)
        # And no extra fields: a leak under a new key name would pass the word check.
        assert {k for b in desk.list_batches() for k in b} == \
            {"batch_id", "table", "rows", "staging_table", "source"}
        for b in desk.list_batches():
            prof = desk.profile_batch(b["batch_id"])
            assert set(prof) == {"batch_id", "table", "rows", "columns"}
            assert {k for col in prof["columns"].values() for k in col} == \
                {"n_null", "n_blank", "n_distinct", "min", "max", "numeric_min", "numeric_max", "top_values"}
        for word in ("defective", "truth", "accepted", c["family"], c["case_id"]):
            assert word not in blob, (c["case_id"], word)
