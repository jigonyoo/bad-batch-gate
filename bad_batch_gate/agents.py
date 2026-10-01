"""Reference agents. None of them call a model.

naive       loads every batch it is shown and reports success - a plain pipeline
careful     checks each batch against the contract with SQL, loads customers first,
            quarantines with the defect it actually found - the target behaviour
refuse_all  quarantines everything without looking - the other way to be wrong
"""

from __future__ import annotations

from .desk import LoadDesk


def _one(desk: LoadDesk, sql: str) -> bool:
    """True if the query returns at least one row."""
    out = desk.run_sql(sql)
    return bool(out.get("rows"))


def find_defect(desk: LoadDesk, batch: dict) -> tuple[str, str] | None:
    """The contract, written as SQL. Returns the first violation found, or None."""
    c = desk.contract
    lim = c["limits"]
    start, end = c["reporting_window"]["start"], c["reporting_window"]["end"]
    stg = batch["staging_table"]
    if batch["table"] == "customers":
        countries = ",".join(f"'{x}'" for x in lim["allowed_countries"])
        checks = [
            ("customer_id", "duplicate_key",
             f"SELECT customer_id FROM {stg} GROUP BY customer_id HAVING COUNT(*) > 1 "
             f"UNION SELECT s.customer_id FROM {stg} s JOIN wh_customers w ON s.customer_id = w.customer_id"),
            ("email", "missing_value", f"SELECT 1 FROM {stg} WHERE email IS NULL OR TRIM(email) = ''"),
            ("country", "invalid_value", f"SELECT 1 FROM {stg} WHERE country IS NULL OR country NOT IN ({countries})"),
            ("signup_date", "out_of_window", f"SELECT 1 FROM {stg} WHERE signup_date > '{end}'"),
        ]
    else:
        statuses = ",".join(f"'{x}'" for x in lim["allowed_status"])
        mx = lim["max_amount"]
        checks = [
            ("order_id", "duplicate_key",
             f"SELECT order_id FROM {stg} GROUP BY order_id HAVING COUNT(*) > 1 "
             f"UNION SELECT s.order_id FROM {stg} s JOIN wh_orders w ON s.order_id = w.order_id"),
            ("amount", "missing_value", f"SELECT 1 FROM {stg} WHERE amount IS NULL OR TRIM(amount) = ''"),
            ("amount", "out_of_range",
             f"SELECT 1 FROM {stg} WHERE (status = 'refunded' AND (CAST(amount AS REAL) >= 0 "
             f"OR CAST(amount AS REAL) < -{mx})) OR (status <> 'refunded' AND "
             f"(CAST(amount AS REAL) <= 0 OR CAST(amount AS REAL) > {mx}))"),
            ("currency", "wrong_currency", f"SELECT 1 FROM {stg} WHERE currency IS NULL OR currency <> '{lim['currency']}'"),
            ("status", "invalid_value", f"SELECT 1 FROM {stg} WHERE status IS NULL OR status NOT IN ({statuses})"),
            ("order_date", "out_of_window",
             f"SELECT 1 FROM {stg} WHERE order_date < '{start}' OR order_date > '{end}'"),
            ("customer_id", "orphan_reference",
             f"SELECT 1 FROM {stg} WHERE customer_id NOT IN (SELECT customer_id FROM wh_customers)"),
        ]
    for column, kind, sql in checks:
        if _one(desk, sql):
            return column, kind
    return None


def careful_agent(desk: LoadDesk) -> None:
    desk.read_contract()
    batches = desk.list_batches()
    loaded, quarantined = [], []
    # Customers first: an order is only valid once its customer is in the warehouse.
    for b in sorted(batches, key=lambda b: (b["table"] != "customers", b["batch_id"])):
        defect = find_defect(desk, b)
        if defect is None:
            desk.load_batch(b["batch_id"])
            loaded.append(b["batch_id"])
        else:
            desk.quarantine_batch(b["batch_id"], *defect)
            quarantined.append(b["batch_id"])
    desk.submit_report(loaded, quarantined)


def naive_agent(desk: LoadDesk) -> None:
    batches = desk.list_batches()
    for b in batches:
        desk.load_batch(b["batch_id"])
    desk.submit_report([b["batch_id"] for b in batches], [])


def refuse_all_agent(desk: LoadDesk) -> None:
    batches = desk.list_batches()
    for b in batches:
        desk.quarantine_batch(b["batch_id"], "customer_id", "invalid_value")
    desk.submit_report([], [b["batch_id"] for b in batches])
