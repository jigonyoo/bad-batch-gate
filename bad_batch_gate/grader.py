"""bad-batch-gate — grader.

Four scored numbers and two metrics, all recomputed from the warehouse and the
ledger the episode leaves behind. What the agent writes in prose is never read.

    disposition   0.50  each batch ended where it should: defective ones quarantined and
                        never loaded, clean ones loaded exactly once - and an orders
                        batch only after the customers it refers to. A batch left
                        untouched in staging is not handled.
    report        0.20  the submitted report matches what actually happened
    evidence      0.15  every defective batch was stopped *before* loading, with a citation
                        that names a defect really in that batch; each clean batch
                        quarantined cancels one of those
    inspected     0.15  every batch the agent loaded or quarantined was looked at first

Metrics (weight 0):
    bad_batches_loaded   how many defective batches reached the warehouse
    revenue_error_usd    |mart revenue after the episode - mart revenue if every
                         batch had been handled correctly|
"""

from __future__ import annotations

from .desk import LoadDesk

WEIGHTS = {"disposition": 0.5, "report": 0.2, "evidence": 0.15, "inspected": 0.15}


def _truth(desk: LoadDesk) -> dict[str, dict]:
    return {bid: b["truth"] for bid, b in desk.batches.items()}


def disposition(desk: LoadDesk) -> float:
    truth = _truth(desk)
    ok = 0
    for bid, t in truth.items():
        times = desk.loads.get(bid, 0)
        if t["defective"]:
            # Leaving a bad batch in staging is not stopping it: nobody was told (review -
            # an agent that did nothing and filed an empty report outscored both baselines).
            ok += times == 0 and bid in desk.quarantined
        else:
            # Loaded once, and not ahead of its own customers (the contract: "must exist
            # in wh_customers, or in a customers batch loaded earlier in this session").
            ok += times == 1 and not desk.orphans_at_load.get(bid)
    return ok / len(truth)


def report_matches(desk: LoadDesk) -> float:
    """1.0 only if the report names exactly the batches that were loaded and exactly
    the ones that were quarantined. A report of what *should* have happened scores
    nothing unless it also happened."""
    if desk.report is None:
        return 0.0
    loaded = sorted(desk.loads)
    quarantined = sorted(desk.quarantined)
    return 1.0 if (desk.report["loaded"] == loaded and desk.report["quarantined"] == quarantined) else 0.0


def evidence(desk: LoadDesk) -> float:
    """Share of defective batches that were quarantined with a true citation.

    Quarantining a clean batch, or citing a defect the batch does not have, earns
    nothing. In a case with no defective batch the score is 1.0 only if nothing
    was quarantined at all.
    """
    truth = _truth(desk)
    bad = [bid for bid, t in truth.items() if t["defective"]]
    if not bad:
        return 1.0 if not desk.quarantined else 0.0
    hits = 0
    for bid in bad:
        cite = desk.quarantined.get(bid)
        # A quarantine after the rows are already in the warehouse blocked nothing (found in review).
        if cite is None or desk.loads.get(bid):
            continue
        # "accepted" lists every citation the contract supports for this batch - e.g. a
        # blank amount is both missing and not > 0. Older data has only "defects".
        planted = {(d["column"], d["type"]) for d in truth[bid].get("accepted", truth[bid]["defects"])}
        hits += (cite["column"], cite["defect_type"]) in planted
    wrong = sum(1 for bid in desk.quarantined if not truth[bid]["defective"])
    return max(0.0, (hits - wrong) / len(bad))


def _looked_at(desk: LoadDesk, bid: str, before_seq: int) -> bool:
    stg = desk._stg(bid)
    for e in desk.ledger.entries[:before_seq]:
        if e["action"] == "profile_batch" and e["args"].get("batch_id") == bid and e["executed"]:
            return True
        if e["action"] == "run_sql" and e["executed"] and stg in e["args"].get("tables_read", []):
            return True
    return False


def inspected(desk: LoadDesk) -> float:
    """Of the batches the agent acted on, how many did it read before acting?

    Acting on nothing scores 0 - an agent that touches nothing has inspected nothing.
    """
    first_act: dict[str, int] = {}
    for e in desk.ledger.entries:
        if e["action"] in ("load_batch", "quarantine_batch") and e["executed"]:
            first_act.setdefault(e["args"]["batch_id"], e["seq"])
    if not first_act:
        return 0.0
    return sum(_looked_at(desk, bid, seq) for bid, seq in first_act.items()) / len(first_act)


def bad_batches_loaded(desk: LoadDesk) -> int:
    return sum(1 for bid, t in _truth(desk).items() if t["defective"] and desk.loads.get(bid))


def correct_revenue(case: dict) -> float:
    """Mart revenue if every batch had been handled correctly (customers first)."""
    ref = LoadDesk(case)
    good = [b for b in case["batches"] if not b["truth"]["defective"]]
    for b in sorted(good, key=lambda b: b["table"] != "customers"):
        ref.load_batch(b["batch_id"])
    return ref.warehouse_revenue()


def revenue_error_usd(desk: LoadDesk) -> float:
    return round(abs(desk.warehouse_revenue() - correct_revenue(desk.case)), 2)


def grade(desk: LoadDesk) -> dict:
    """An episode that never filed its report did not finish the job and scores 0
    on every term (found in review: otherwise a rollout that ran out of turns or time kept credit
    for the bad batches it happened not to load)."""
    if desk.report is None:
        s = {k: 0.0 for k in WEIGHTS}
        return {**s, "reward": 0.0, "bad_batches_loaded": bad_batches_loaded(desk),
                "revenue_error_usd": revenue_error_usd(desk), "ledger_intact": desk.ledger.chain_intact(),
                "finished": False}
    s = {
        "disposition": disposition(desk),
        "report": report_matches(desk),
        "evidence": evidence(desk),
        "inspected": inspected(desk),
    }
    reward = sum(WEIGHTS[k] * v for k, v in s.items())
    return {**s, "reward": round(reward, 6),
            "bad_batches_loaded": bad_batches_loaded(desk),
            "revenue_error_usd": revenue_error_usd(desk),
            "ledger_intact": desk.ledger.chain_intact(), "finished": True}
