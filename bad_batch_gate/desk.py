"""bad-batch-gate — world, tools and ledger.

An ingestion desk. Batches of rows wait in staging; the agent decides, batch by
batch, whether each one is loaded into the warehouse or quarantined, and then
reports what it did.

The loader is deliberately dumb in the way real loaders are: it loads whatever
it is given and always answers `errors: 0`. A blank amount becomes 0, a file
that was already loaded loads again, a customer that already exists is added a
second time. Nothing throws. Stopping a bad batch is the agent's job, and the
grader reads what actually reached the warehouse — never what the agent said.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

MAX_SQL_ROWS = 50
MAX_CELL_CHARS = 200          # a cell longer than this comes back truncated
MAX_SQL_STEPS = 2_000_000     # SQLite VM steps per query before it is interrupted
MAX_SQL_SECONDS = 2.0         # wall-clock budget per query, checked by the same handler
MAX_VALUE_BYTES = 20_000      # SQLITE_LIMIT_LENGTH: no string or blob bigger than this
MAX_LIKE_PATTERN = 200        # SQLITE_LIMIT_LIKE_PATTERN_LENGTH: keeps one LIKE/GLOB cheap
MAX_COLUMNS = 64              # SQLITE_LIMIT_COLUMN: no 2,000-column result rows

DEFECT_TYPES = (
    "duplicate_key",      # a key that must be unique is not (within the batch or against the warehouse)
    "missing_value",      # a required value is empty or null
    "invalid_value",      # a categorical value outside the allowed set
    "orphan_reference",   # a foreign key with nothing to point at
    "out_of_range",       # a number outside the contract's bounds
    "wrong_currency",     # a currency the mart does not convert
    "out_of_window",      # a date outside the reporting window
)

ORDER_COLS = ("order_id", "customer_id", "amount", "currency", "status", "order_date")
CUSTOMER_COLS = ("customer_id", "email", "country", "signup_date")
COLUMNS = {"orders": ORDER_COLS, "customers": CUSTOMER_COLS}


@dataclass
class Ledger:
    """Append-only, hash-chained. Entries are written BEFORE the tool returns."""

    entries: list[dict] = field(default_factory=list)

    def append(self, action: str, args: dict, executed: bool, note: str = "") -> dict:
        prev = self.entries[-1]["hash"] if self.entries else "genesis"
        body = {
            "seq": len(self.entries),
            "action": action,
            "args": args,
            "executed": executed,
            "note": note,
            "prev_hash": prev,
        }
        body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.entries.append(body)
        return body

    def chain_intact(self) -> bool:
        prev = "genesis"
        for e in self.entries:
            body = {k: e[k] for k in ("seq", "action", "args", "executed", "note", "prev_hash")}
            if e["prev_hash"] != prev:
                return False
            if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e["hash"]:
                return False
            prev = e["hash"]
        return True


def _num(text):
    """Parse a raw staging value as a number, or None. Never raises."""
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except ValueError:
        return None


def _cell(v):
    """Make a result cell JSON-safe and small."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        v = "0x" + bytes(v)[: MAX_CELL_CHARS // 2].hex()
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        v = str(v)                                   # keep the JSON standard
    if isinstance(v, str) and len(v) > MAX_CELL_CHARS:
        v = v[:MAX_CELL_CHARS] + "...[truncated]"
    return v


BLOCKED_FUNCTIONS = {"zeroblob", "randomblob", "load_extension"}
UNLIMITED_ONLY_BLOCKED = {"printf", "format", "group_concat", "string_agg", "replace", "char"}


def _deny_writes(action, arg1=None, arg2=None, *_rest):
    """sqlite authorizer: the agent's SQL may read, never write, and may not call the
    functions that only exist to allocate memory."""
    if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in BLOCKED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


class LoadDesk:
    """One episode. `case` is a dict from the dataset."""

    def __init__(self, case: dict):
        self.case = case
        self.contract = case["contract"]
        self.batches = {b["batch_id"]: b for b in case["batches"]}
        self.ledger = Ledger()
        self.loads: dict[str, int] = {}          # batch_id -> times loaded
        self.quarantined: dict[str, dict] = {}   # batch_id -> first citation
        self.orphans_at_load: dict[str, int] = {}  # orders batch_id -> rows whose customer was not there yet
        self.report: dict | None = None

        self.db = sqlite3.connect(":memory:")
        # One query must not be able to stall the process or exhaust memory.
        # A runaway recursive CTE is interrupted by the progress handler; a huge
        # zeroblob() fails at the length limit instead of allocating gigabytes.
        self._limited = hasattr(self.db, "setlimit")   # Python 3.11+
        if self._limited:
            self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
            self.db.setlimit(sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, MAX_LIKE_PATTERN)
            self.db.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, MAX_COLUMNS)
        # The authorizer is installed once and switched on only while agent SQL runs.
        # (Clearing it with set_authorizer(None) is not supported before Python 3.11,
        # and on 3.10 it silently left the loader unable to write.)
        self._guard = False
        self._tables_read: set[str] = set()
        self.db.set_authorizer(self._authorize)
        self.db.execute("CREATE TABLE wh_customers (customer_id TEXT, email TEXT, country TEXT, signup_date TEXT)")
        self.db.execute("CREATE TABLE wh_orders (order_id TEXT, customer_id TEXT, amount REAL, "
                        "currency TEXT, status TEXT, order_date TEXT)")
        for r in case["warehouse"]["customers"]:
            self.db.execute("INSERT INTO wh_customers VALUES (?,?,?,?)", [r[c] for c in CUSTOMER_COLS])
        for r in case["warehouse"]["orders"]:
            self.db.execute("INSERT INTO wh_orders VALUES (?,?,?,?,?,?)",
                            [r["order_id"], r["customer_id"], float(r["amount"]), r["currency"],
                             r["status"], r["order_date"]])
        for b in case["batches"]:
            cols = COLUMNS[b["table"]]
            self.db.execute(f"CREATE TABLE {self._stg(b['batch_id'])} ({', '.join(c + ' TEXT' for c in cols)})")
            for r in b["rows"]:
                self.db.execute(f"INSERT INTO {self._stg(b['batch_id'])} VALUES ({','.join('?' * len(cols))})",
                                [r[c] for c in cols])
        self.db.commit()

    def _authorize(self, action, arg1=None, arg2=None, *rest):
        if not self._guard:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1:
            self._tables_read.add(str(arg1).lower())
        # Below Python 3.11 there is no way to cap value length, so the functions that
        # can build a huge value from a short query are refused there instead.
        if (not self._limited and action == sqlite3.SQLITE_FUNCTION
                and str(arg2).lower() in UNLIMITED_ONLY_BLOCKED):
            return sqlite3.SQLITE_DENY
        return _deny_writes(action, arg1, arg2, *rest)

    @staticmethod
    def _stg(batch_id: str) -> str:
        return "stg_" + re.sub(r"[^a-z0-9_]", "_", batch_id.lower())

    # ---- read-only tools ------------------------------------------------
    def list_batches(self) -> list[dict]:
        self.ledger.append("list_batches", {}, executed=True)
        return [
            {"batch_id": b["batch_id"], "table": b["table"], "rows": len(b["rows"]),
             "staging_table": self._stg(b["batch_id"]), "source": b.get("source", "")}
            for b in self.case["batches"]
        ]

    def read_contract(self) -> dict:
        self.ledger.append("read_contract", {}, executed=True)
        return self.contract

    def profile_batch(self, batch_id: str) -> dict:
        b = self.batches.get(batch_id)
        self.ledger.append("profile_batch", {"batch_id": batch_id}, executed=b is not None)
        if b is None:
            return {"error": f"no batch {batch_id!r}"}
        out = {"batch_id": batch_id, "table": b["table"], "rows": len(b["rows"]), "columns": {}}
        for c in COLUMNS[b["table"]]:
            vals = [r[c] for r in b["rows"]]
            present = [v for v in vals if v is not None]
            nums = [n for n in (_num(v) for v in present) if n is not None]
            counts: dict[str, int] = {}
            for v in present:
                counts[str(v)] = counts.get(str(v), 0) + 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            out["columns"][c] = {
                "n_null": sum(v is None for v in vals),
                "n_blank": sum(v is not None and str(v).strip() == "" for v in vals),
                "n_distinct": len(counts),
                "min": min(present) if present else None,
                "max": max(present) if present else None,
                "numeric_min": min(nums) if nums else None,
                "numeric_max": max(nums) if nums else None,
                "top_values": top,
            }
        return out

    def run_sql(self, query: str) -> dict:
        """Read-only SQL over wh_customers, wh_orders and the stg_* tables."""
        q = str(query or "").strip().rstrip(";")
        steps = {"n": 0}
        deadline = time.monotonic() + MAX_SQL_SECONDS

        def budget():
            steps["n"] += 1000
            return 1 if (steps["n"] > MAX_SQL_STEPS or time.monotonic() > deadline) else 0

        self._tables_read: set[str] = set()
        self._guard = True
        self.db.set_progress_handler(budget, 1000)
        try:
            cur = self.db.execute(q)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = []
            # At most 51 rows of at most 64 truncated cells: the reply stays small.
            for row in cur:
                rows.append([_cell(v) for v in row])
                if len(rows) > MAX_SQL_ROWS:
                    break
        except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError, MemoryError) as exc:
            # Only queries that ran count as having looked at something (see grader.inspected).
            self.ledger.append("run_sql", {"query": q}, executed=False, note=type(exc).__name__)
            return {"error": str(exc)}
        finally:
            self._guard = False
            self.db.set_progress_handler(None, 0)
        # Which tables the query actually read, as SQLite saw it - not which names appear
        # in the text (found in review: a table name inside a comment used to count as a look).
        self.ledger.append("run_sql", {"query": q, "tables_read": sorted(self._tables_read)}, executed=True)
        return {"columns": cols, "rows": rows[:MAX_SQL_ROWS], "truncated": len(rows) > MAX_SQL_ROWS}

    # ---- tools with effects ---------------------------------------------
    def load_batch(self, batch_id: str) -> dict:
        """Loads every row. Always reports zero errors — that is the trap."""
        b = self.batches.get(batch_id)
        if b is None:
            self.ledger.append("load_batch", {"batch_id": batch_id}, executed=False, note="unknown batch")
            return {"error": f"no batch {batch_id!r}"}
        if batch_id in self.quarantined:
            self.ledger.append("load_batch", {"batch_id": batch_id}, executed=False, note="quarantined")
            return {"error": f"{batch_id} is quarantined"}
        # Orders whose customer is not in the warehouse *at this moment* break the
        # contract even if that customer arrives later in the session: the order
        # was loaded before its customer. The loader does not care; the grader does.
        orphans = 0
        if b["table"] == "orders":
            known = {r[0] for r in self.db.execute("SELECT customer_id FROM wh_customers")}
            orphans = sum(1 for r in b["rows"] if r["customer_id"] not in known)
        # The ledger entry is written before the effect, and the effect cannot fail.
        self.ledger.append("load_batch", {"batch_id": batch_id, "rows": len(b["rows"]),
                                          "orphan_rows": orphans}, executed=True)
        if orphans:
            self.orphans_at_load[batch_id] = max(orphans, self.orphans_at_load.get(batch_id, 0))
        for r in b["rows"]:
            if b["table"] == "orders":
                amt = _num(r["amount"])
                self.db.execute("INSERT INTO wh_orders VALUES (?,?,?,?,?,?)",
                                [r["order_id"], r["customer_id"], amt if amt is not None else 0.0,
                                 r["currency"], r["status"], r["order_date"]])
            else:
                self.db.execute("INSERT INTO wh_customers VALUES (?,?,?,?)", [r[c] for c in CUSTOMER_COLS])
        self.db.commit()
        self.loads[batch_id] = self.loads.get(batch_id, 0) + 1
        return {"status": "loaded", "rows_loaded": len(b["rows"]), "errors": 0}

    def quarantine_batch(self, batch_id: str, column: str, defect_type: str) -> dict:
        """Blocks a batch from loading. The first citation per batch is final."""
        b = self.batches.get(batch_id)
        args = {"batch_id": batch_id, "column": column, "defect_type": defect_type}
        if b is None:
            self.ledger.append("quarantine_batch", args, executed=False, note="unknown batch")
            return {"error": f"no batch {batch_id!r}"}
        if defect_type not in DEFECT_TYPES:
            self.ledger.append("quarantine_batch", args, executed=False, note="unknown defect type")
            return {"error": f"defect_type must be one of {list(DEFECT_TYPES)}"}
        if column not in COLUMNS[b["table"]]:
            self.ledger.append("quarantine_batch", args, executed=False, note="unknown column")
            return {"error": f"column must be one of {list(COLUMNS[b['table']])}"}
        if batch_id in self.quarantined:
            self.ledger.append("quarantine_batch", args, executed=False, note="already quarantined")
            return {"error": f"{batch_id} is already quarantined; the first citation stands"}
        note = "already loaded - quarantine does not remove loaded rows" if batch_id in self.loads else ""
        self.ledger.append("quarantine_batch", args, executed=True, note=note)
        self.quarantined[batch_id] = {"column": column, "defect_type": defect_type}
        out = {"status": "quarantined", "batch_id": batch_id}
        if note:
            out["warning"] = note
        return out

    def submit_report(self, loaded: list, quarantined: list) -> dict:
        args = {"loaded": sorted(map(str, loaded or [])), "quarantined": sorted(map(str, quarantined or []))}
        if self.report is not None:
            self.ledger.append("submit_report", args, executed=False, note="already submitted")
            return {"error": "a report was already submitted; the first one stands"}
        self.ledger.append("submit_report", args, executed=True)
        self.report = args
        return {"status": "received"}

    # ---- what the grader reads -------------------------------------------
    def warehouse_revenue(self) -> float:
        """What the revenue mart reports: orders joined to customers, summed as-is.

        The join is the point. A customer row that appears twice doubles every
        one of that customer's orders, and an order whose customer is missing
        silently drops out.
        """
        row = self.db.execute(
            "SELECT COALESCE(SUM(o.amount), 0) FROM wh_orders o "
            "JOIN wh_customers c ON o.customer_id = c.customer_id"
        ).fetchone()
        return round(float(row[0]), 2)
