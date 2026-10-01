"""bad-batch-gate — dataset.

Every case is generated from an explicit spec with a fixed seed, so the eval
split is reproducible byte for byte and every planted defect is listed next to
the batch it lives in. The grader reads those truth labels; the agent never
sees them.

Eight families. Six plant exactly one kind of defect in one batch; two are
controls that contain none, and one of those is built to look suspicious.

    clean-control      nothing wrong; load everything
    resent-rows        order ids that already exist (a re-sent file) or repeat inside the batch
    amount-bounds      a cents/dollars mix-up, or a negative amount on a non-refund
    blank-amount       an empty amount, which the loader turns into 0
    orphan-customer    an order for a customer who does not exist
    label-drift        'Delivered' vs 'delivered', EUR in a USD mart, a date outside the window
    dimension-fanout   a customers batch that re-adds an existing customer id
    legit-edge         refunds, an amount exactly at the cap, a date on the window's last day,
                       orders for customers that arrive in the same session - all valid
"""

from __future__ import annotations

import json
import random
from pathlib import Path

WINDOW = ("2026-09-01", "2026-09-30")
MAX_AMOUNT = 50000
ALLOWED_STATUS = ["placed", "shipped", "delivered", "returned", "refunded"]
ALLOWED_COUNTRIES = ["US", "CA", "GB", "DE", "FR", "KR", "JP", "AU"]

FAMILIES = [
    "clean-control", "resent-rows", "amount-bounds", "blank-amount",
    "orphan-customer", "label-drift", "dimension-fanout", "legit-edge",
]
CONTROL_FAMILIES = {"clean-control", "legit-edge"}

CONTRACT = {
    "reporting_window": {"start": WINDOW[0], "end": WINDOW[1]},
    "orders": {
        "order_id": "unique across the warehouse and the batch",
        "customer_id": "must exist in wh_customers, or in a customers batch loaded earlier in this session",
        "amount": f"required number; 0 < amount <= {MAX_AMOUNT}, except status 'refunded' "
                  f"where -{MAX_AMOUNT} <= amount < 0",
        "currency": "'USD' only - the revenue mart sums amounts without conversion",
        "status": "exactly one of " + ", ".join(ALLOWED_STATUS) + " (lower case, as written)",
        "order_date": "YYYY-MM-DD inside the reporting window",
    },
    "customers": {
        "customer_id": "unique across the warehouse and the batch",
        "email": "required, not blank",
        "country": "ISO-2, one of " + ", ".join(ALLOWED_COUNTRIES),
        "signup_date": "YYYY-MM-DD, not after the end of the reporting window",
    },
    "limits": {"max_amount": MAX_AMOUNT, "currency": "USD", "allowed_status": ALLOWED_STATUS,
               "allowed_countries": ALLOWED_COUNTRIES},
    "defect_types": ["duplicate_key", "missing_value", "invalid_value", "orphan_reference",
                     "out_of_range", "wrong_currency", "out_of_window"],
}


def _date(rng: random.Random, lo: int = 1, hi: int = 30) -> str:
    return f"2026-09-{rng.randint(lo, hi):02d}"


def _money(rng: random.Random, lo: float = 12.0, hi: float = 1900.0) -> str:
    return f"{rng.uniform(lo, hi):.2f}"


class _Ids:
    """Ids are drawn at random, not counted up. Counting leaked the answer: the
    defective batch is built first, so its order ids were always the smallest
    (third review: sorting by id alone scored 0.825)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def _draw(self, prefix: str, lo: int, hi: int) -> str:
        while True:
            v = f"{prefix}{self.rng.randint(lo, hi)}"
            if v not in self.used:
                self.used.add(v)
                return v

    def order(self) -> str:
        return self._draw("O", 10000, 99999)

    def customer(self) -> str:
        return self._draw("C", 100, 899)


def _warehouse(rng: random.Random, ids: _Ids) -> dict:
    # Warehouse, new and orphan customer ids all come from one random range, and
    # warehouse and batch orders share one date range, so neither an id nor a date
    # says where a row came from (fourth review: min date and id range alone
    # identified re-sent rows, orphans and re-added customers - 0.838).
    customers = []
    for _ in range(20):
        cid = ids.customer()
        customers.append({"customer_id": cid, "email": f"{cid.lower()}@example.com",
                          "country": rng.choice(ALLOWED_COUNTRIES),
                          "signup_date": f"2026-0{rng.randint(1, 9)}-{rng.randint(1, 28):02d}"})
    orders = [{"order_id": ids.order(), "customer_id": rng.choice(customers)["customer_id"],
               "amount": _money(rng), "currency": "USD",
               "status": rng.choice(ALLOWED_STATUS[:4]), "order_date": _date(rng, 1, 30)}
              for _ in range(30)]
    return {"customers": customers, "orders": orders}


def _clean_orders(rng, ids, customer_ids, n=None) -> list[dict]:
    n = n or rng.randint(12, 20)
    return [{"order_id": ids.order(), "customer_id": rng.choice(customer_ids),
             "amount": _money(rng), "currency": "USD",
             "status": rng.choice(ALLOWED_STATUS[:4]), "order_date": _date(rng, 1, 30)}
            for _ in range(n)]


def _clean_customers(rng, ids, n=None) -> list[dict]:
    n = n or rng.randint(3, 6)
    out = []
    for _ in range(n):
        cid = ids.customer()
        out.append({"customer_id": cid, "email": f"{cid.lower()}@example.com",
                    "country": rng.choice(ALLOWED_COUNTRIES),
                    "signup_date": f"2026-0{rng.randint(1, 9)}-{rng.randint(1, 28):02d}"})
    return out


def _batch(batch_id, table, rows, source, defects=(), also=()) -> dict:
    """`defects` is what was planted. `also` lists further citations the contract
    supports for the same rows; the grader accepts any of them."""
    truth = {"defective": bool(defects), "defects": [{"column": c, "type": t} for c, t in defects]}
    if also:
        truth["accepted"] = truth["defects"] + [{"column": c, "type": t} for c, t in also]
    return {"batch_id": batch_id, "table": table, "rows": rows, "source": source, "truth": truth}


def build_case(family: str, variant: int, seed: int, case_id: str) -> dict:
    rng = random.Random(seed)
    ids = _Ids(rng)
    wh = _warehouse(rng, ids)
    wh_cids = [c["customer_id"] for c in wh["customers"]]
    tag = f"0930-{rng.randint(10, 99)}"
    batches: list[dict] = []

    def orders_batch(rows, defects=(), also=()):
        # Source notes are drawn independently of the defect, so they carry no signal
        # (a review found "(retried after timeout)" used to appear only on re-sent files).
        src = rng.choice(["shop-api export", "pos export", "shop-api export (retry)"])
        batches.append(_batch("pending", "orders", rows, src, defects, also))

    def pick(rows, need=lambda r: True):
        cand = [i for i, r in enumerate(rows) if need(r)]
        return rng.choice(cand)

    def new_customers_with_orders(rows):
        # New customers and their first orders arrive together. The orders are valid
        # only once the customers batch has been loaded first.
        new = _clean_customers(rng, ids)
        batches.append(_batch("pending", "customers", new, "crm sync"))
        for j in rng.sample(range(len(rows)), 3):
            rows[j]["customer_id"] = rng.choice(new)["customer_id"]

    # Every case has exactly two batches, and a clean customers batch appears as often
    # as a defective one, so neither the batch count, the table nor the listing position
    # predicts the answer (second review: an agent that never read a row scored 0.744
    # on those priors alone).
    if family == "clean-control":
        rows = _clean_orders(rng, ids, wh_cids)
        if variant in (1, 3):
            new_customers_with_orders(rows)
            orders_batch(rows)
        else:
            orders_batch(rows)
            orders_batch(_clean_orders(rng, ids, wh_cids))

    elif family == "resent-rows":
        rows = _clean_orders(rng, ids, wh_cids)
        if variant in (0, 1):
            # A re-sent file: some rows are orders the warehouse already holds.
            # Nothing inside the batch repeats, so a profile of the batch alone looks clean.
            for i in rng.sample(range(len(rows)), 3):
                rows[i] = dict(rng.choice(wh["orders"]))
        else:
            # The same order twice inside the batch. It replaces another row rather than
            # being added, so the row count gives nothing away.
            i, j = rng.sample(range(len(rows)), 2)
            rows[j] = dict(rows[i])
        orders_batch(rows, [("order_id", "duplicate_key")])

    elif family == "amount-bounds":
        rows = _clean_orders(rng, ids, wh_cids)
        i = pick(rows)
        if variant in (0, 2):
            rows[i]["amount"] = f"{rng.randint(900, 4800) * 100:.2f}"     # cents read as dollars
        else:
            rows[i]["amount"] = f"-{rng.uniform(20, 600):.2f}"             # negative, not a refund
            rows[i]["status"] = rng.choice(["placed", "shipped"])
        orders_batch(rows, [("amount", "out_of_range")])

    elif family == "blank-amount":
        rows = _clean_orders(rng, ids, wh_cids)
        rows[pick(rows)]["amount"] = "" if variant < 2 else "   "
        # A blank amount is missing, and it also fails "0 < amount" once the loader
        # turns it into 0. The contract supports both readings, so both count (a ceiling review).
        orders_batch(rows, [("amount", "missing_value")], also=[("amount", "out_of_range")])

    elif family == "orphan-customer":
        rows = _clean_orders(rng, ids, wh_cids)
        rows[pick(rows)]["customer_id"] = ids.customer()                # same range, just nobody's
        orders_batch(rows, [("customer_id", "orphan_reference")])

    elif family == "label-drift":
        rows = _clean_orders(rng, ids, wh_cids)
        i = pick(rows)
        if variant == 0:
            rows[i]["status"] = "Delivered"
            orders_batch(rows, [("status", "invalid_value")])
        elif variant == 1:
            for j in rng.sample(range(len(rows)), 2):
                rows[j]["currency"] = "EUR"
            orders_batch(rows, [("currency", "wrong_currency")])
        elif variant == 2:
            rows[i]["order_date"] = "2026-08-31"
            orders_batch(rows, [("order_date", "out_of_window")])
        else:
            rows[i]["status"] = "shipped "        # trailing space: not "exactly one of" the allowed values
            orders_batch(rows, [("status", "invalid_value")])

    elif family == "dimension-fanout":
        new = _clean_customers(rng, ids)
        # An exact re-send of a customer the warehouse already holds. Nothing about the
        # row itself differs (fifth review: a "+crm" email and an older signup date
        # used to mark it - 44/44).
        dup = dict(rng.choice(wh["customers"]))
        new[rng.randrange(len(new))] = dup                            # replaces, so the count is typical
        batches.append(_batch("pending", "customers", new, "crm sync",
                              [("customer_id", "duplicate_key")]))
        # A clean orders batch that includes the duplicated customer: once both load,
        # every one of that customer's orders is counted twice in the mart.
        rows = _clean_orders(rng, ids, wh_cids)
        rows[0]["customer_id"] = dup["customer_id"]
        orders_batch(rows)

    elif family == "legit-edge":
        rows = _clean_orders(rng, ids, wh_cids)
        if variant == 0:
            i = pick(rows)
            rows[i]["status"], rows[i]["amount"] = "refunded", f"-{rng.uniform(20, 400):.2f}"
            orders_batch(rows)
            orders_batch(_clean_orders(rng, ids, wh_cids))
        elif variant == 1:
            rows[pick(rows)]["amount"] = f"{MAX_AMOUNT:.2f}"
            rows[pick(rows)]["order_date"] = WINDOW[1]
            orders_batch(rows)
            orders_batch(_clean_orders(rng, ids, wh_cids))
        else:
            new_customers_with_orders(rows)
            if variant == 3:
                rows[pick(rows)]["status"] = "returned"
            orders_batch(rows)
    else:
        raise ValueError(family)

    if family not in CONTROL_FAMILIES and family != "dimension-fanout":
        # A clean batch next to the bad one: stopping everything is not the answer.
        orders_batch(_clean_orders(rng, ids, wh_cids))

    if (family, variant) in (("legit-edge", 2), ("clean-control", 3)):
        # Files arrive in whatever order they arrive. Listing the orders before the
        # customers they depend on is the point of this variant: an agent that checks
        # in listed order sees valid orders as orphans (without this, an agent
        # that ignored load order scored a perfect 1.0).
        batches.sort(key=lambda b: b["table"] != "orders")
    else:
        rng.shuffle(batches)
    # Ids are assigned in listed order, after shuffling, so the suffix says nothing
    # about which batch is bad (the defective batch used to be "-1" every time).
    for k, b in enumerate(batches, start=1):
        b["batch_id"] = f"B-{tag}-{k}"
    n = len(batches)
    # The run date is after the window closes, so no valid order is "in the future"
    # (an ingest dated 09-26 next to orders dated 09-30 was a trap the contract never set).
    prompt = (f"September close, run on 2026-10-01. {n} batch{'es' if n > 1 else ''} waiting in staging. "
              "Load what meets the data contract, quarantine what does not, then submit your report.")
    return {"case_id": case_id, "family": family, "variant": variant, "seed": seed,
            "prompt": prompt, "contract": CONTRACT, "warehouse": wh, "batches": batches}


def build_eval() -> list[dict]:
    cases = []
    for fi, fam in enumerate(FAMILIES):
        for v in range(4):
            cases.append(build_case(fam, v, seed=20260930 + fi * 100 + v, case_id=f"eval-{fam}-{v}"))
    return cases


def build_train(n: int = 320, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for k in range(n):
        fam = FAMILIES[k % len(FAMILIES)]
        cases.append(build_case(fam, rng.randint(0, 3), seed=rng.randint(10**6, 10**9),
                                case_id=f"train-{k:03d}"))
    return cases


def write(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, cases in (("eval_curated.jsonl", build_eval()), ("train_procedural.jsonl", build_train())):
        with open(data_dir / name, "w", encoding="utf-8", newline="\n") as f:
            for c in cases:
                f.write(json.dumps(c, sort_keys=True) + "\n")


if __name__ == "__main__":
    write(Path(__file__).parent / "data")
