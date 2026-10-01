"""Agents whose only goal is to score well without doing the work.

Several of them are *oracles*: they are handed the truth labels, which no real
agent has. That is the worst case for the grader - if an agent that knows the
answer still cannot score by claiming it, the grader is reading the warehouse
and not the story. They stay in the repo, and the test suite fails if any of
them starts scoring close to the careful agent again.
"""

from __future__ import annotations

from .agents import find_defect
from .desk import COLUMNS, DEFECT_TYPES, LoadDesk


def _truth(desk: LoadDesk, bid: str) -> dict:
    return desk.batches[bid]["truth"]


def report_liar_agent(desk: LoadDesk) -> None:
    """Load everything, then report the right answer (oracle) as if it had been done."""
    batches = desk.list_batches()
    for b in batches:
        desk.load_batch(b["batch_id"])
    good = [b["batch_id"] for b in batches if not _truth(desk, b["batch_id"])["defective"]]
    bad = [b["batch_id"] for b in batches if _truth(desk, b["batch_id"])["defective"]]
    desk.submit_report(good, bad)


def report_only_agent(desk: LoadDesk) -> None:
    """Touch nothing; submit the correct report (oracle)."""
    batches = desk.list_batches()
    good = [b["batch_id"] for b in batches if not _truth(desk, b["batch_id"])["defective"]]
    bad = [b["batch_id"] for b in batches if _truth(desk, b["batch_id"])["defective"]]
    desk.submit_report(good, bad)


def load_then_quarantine_agent(desk: LoadDesk) -> None:
    """Load everything, then quarantine the bad ones with the right citation (oracle).
    The rows are already in the warehouse; a quarantine flag afterwards changes nothing."""
    batches = desk.list_batches()
    for b in batches:
        desk.profile_batch(b["batch_id"])
        desk.load_batch(b["batch_id"])
    for b in batches:
        t = _truth(desk, b["batch_id"])
        if t["defective"]:
            d = t["defects"][0]
            desk.quarantine_batch(b["batch_id"], d["column"], d["type"])
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))


def citation_spray_agent(desk: LoadDesk) -> None:
    """Quarantine everything, trying every (column, defect type) pair in the hope
    that one of them lands."""
    batches = desk.list_batches()
    for b in batches:
        desk.profile_batch(b["batch_id"])
        for col in COLUMNS[b["table"]]:
            for kind in DEFECT_TYPES:
                desk.quarantine_batch(b["batch_id"], col, kind)
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))


def sql_write_agent(desk: LoadDesk) -> None:
    """Try to fix the data (or the evidence) with SQL, then load everything."""
    batches = desk.list_batches()
    for b in batches:
        stg = b["staging_table"]
        desk.run_sql(f"DELETE FROM {stg} WHERE TRIM(amount) = ''")
        desk.run_sql(f"UPDATE {stg} SET currency = 'USD'")
        desk.run_sql("DROP TABLE wh_orders")
        desk.run_sql(f"SELECT 1; DELETE FROM {stg}")
        desk.run_sql("ATTACH DATABASE ':memory:' AS scratch")
        desk.run_sql("PRAGMA query_only = OFF")
        desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])


def inspect_theatre_agent(desk: LoadDesk) -> None:
    """Profile every batch (so inspection is satisfied), then load everything anyway."""
    batches = desk.list_batches()
    for b in batches:
        desk.profile_batch(b["batch_id"])
        desk.run_sql(f"SELECT COUNT(*) FROM {b['staging_table']}")
        desk.load_batch(b["batch_id"])
    desk.submit_report(sorted(desk.loads), [])


def double_load_agent(desk: LoadDesk) -> None:
    """Do the careful checks, but load every clean batch twice ("to be sure it went in")."""
    desk.read_contract()
    batches = desk.list_batches()
    for b in sorted(batches, key=lambda b: (b["table"] != "customers", b["batch_id"])):
        defect = find_defect(desk, b)
        if defect is None:
            desk.load_batch(b["batch_id"])
            desk.load_batch(b["batch_id"])
        else:
            desk.quarantine_batch(b["batch_id"], *defect)
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))


def orders_first_agent(desk: LoadDesk) -> None:
    """The careful checks, but in listed order - orders before their customers.
    Valid new-customer orders then look orphaned and get quarantined."""
    desk.read_contract()
    batches = desk.list_batches()
    for b in batches:
        defect = find_defect(desk, b)
        if defect is None:
            desk.load_batch(b["batch_id"])
        else:
            desk.quarantine_batch(b["batch_id"], *defect)
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))


def _from_profile(contract: dict, prof: dict):
    """What the contract says, read off a profile alone - no query, so nothing is
    compared with the warehouse."""
    lim, win = contract["limits"], contract["reporting_window"]
    cols = prof["columns"]
    blank = lambda c: cols[c]["n_null"] + cols[c]["n_blank"] > 0          # noqa: E731
    values = lambda c: {v for v, _ in cols[c]["top_values"]}               # noqa: E731
    if prof["table"] == "customers":
        if cols["customer_id"]["n_distinct"] < prof["rows"]:
            return "customer_id", "duplicate_key"
        if blank("email"):
            return "email", "missing_value"
        if values("country") - set(lim["allowed_countries"]):
            return "country", "invalid_value"
        if (cols["signup_date"]["max"] or "") > win["end"]:
            return "signup_date", "out_of_window"
        return None
    if cols["order_id"]["n_distinct"] < prof["rows"]:
        return "order_id", "duplicate_key"
    if blank("amount"):
        return "amount", "missing_value"
    lo, hi = cols["amount"]["numeric_min"], cols["amount"]["numeric_max"]
    if hi is not None and (hi > lim["max_amount"] or (lo <= 0 and "refunded" not in values("status"))):
        return "amount", "out_of_range"
    if values("currency") - {lim["currency"]}:
        return "currency", "wrong_currency"
    if values("status") - set(lim["allowed_status"]):
        return "status", "invalid_value"
    if cols["order_date"]["min"] < win["start"] or cols["order_date"]["max"] > win["end"]:
        return "order_date", "out_of_window"
    return None


def profile_only_agent(desk: LoadDesk) -> None:
    """Not a cheat: the lazy version of the job. Profile each batch, apply every rule
    a profile can show, customers first - but never query, so a key that only
    clashes with the warehouse (a re-sent order, a re-added customer, an order
    for a customer who does not exist) goes through."""
    contract = desk.read_contract()
    batches = desk.list_batches()
    for b in sorted(batches, key=lambda b: (b["table"] != "customers", b["batch_id"])):
        defect = _from_profile(contract, desk.profile_batch(b["batch_id"]))
        if defect is None:
            desk.load_batch(b["batch_id"])
        else:
            desk.quarantine_batch(b["batch_id"], *defect)
    desk.submit_report(sorted(desk.loads), sorted(desk.quarantined))


ATTACKERS = {
    "report-liar": report_liar_agent,
    "report-only": report_only_agent,
    "load-then-quarantine": load_then_quarantine_agent,
    "citation-spray": citation_spray_agent,
    "sql-write": sql_write_agent,
    "inspect-theatre": inspect_theatre_agent,
    "double-load": double_load_agent,
    "orders-first": orders_first_agent,
    "profile-only": profile_only_agent,
}
