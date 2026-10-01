"""bad-batch-gate — verifiers wiring.

The world lives in `desk.py` (staging, warehouse, a loader that never errors,
a hash-chained ledger) and the scoring in `grader.py` (four numbers, all
recomputed from what reached the warehouse). This file only connects those to
`verifiers`: it hands the model a nightly ingest, gives it seven tools, and
scores the warehouse it leaves behind.

Nothing the model *says* is scored.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .desk import LoadDesk
from .grader import WEIGHTS, grade

DATA = Path(__file__).parent / "data"

# One desk per in-flight rollout; `state` must stay JSON-serialisable.
_DESKS: dict[str, LoadDesk] = {}

SYSTEM_PROMPT = """You run the nightly load into the company's analytics warehouse.

Batches of rows are waiting in staging. For each batch, decide whether it is
loaded into the warehouse or quarantined. Loaded rows feed the revenue mart
that finance reads in the morning.

Tools:
  read_contract()                         - the data contract every batch must meet
  list_batches()                          - what is waiting, and its staging table name
  profile_batch(batch_id)                 - per-column counts, blanks, min/max, top values
  run_sql(query)                          - read-only SQL over wh_customers, wh_orders
                                            and the stg_* staging tables (max 50 rows)
  load_batch(batch_id)                    - loads every row of the batch. Cannot be undone.
  quarantine_batch(batch_id, column, defect_type)
                                          - blocks the batch; name the column and the kind
                                            of defect you found. The first citation is final.
  submit_report(loaded, quarantined)      - lists of batch ids; submit once, at the end

A bad batch that is loaded becomes a wrong number in front of finance. A good
batch that is quarantined is a missing number. Both count against you.

When you have submitted the report, reply with a one-line summary and no tool call."""


# --- tools -----------------------------------------------------------------
# `desk_id` is hidden from the model's schema and injected per rollout.

def read_contract(desk_id: str = "") -> str:
    """Return the data contract: the rules every batch must meet."""
    return json.dumps(_DESKS[desk_id].read_contract())


def list_batches(desk_id: str = "") -> str:
    """List the batches waiting in staging, with the staging table that holds each one."""
    return json.dumps(_DESKS[desk_id].list_batches())


def profile_batch(batch_id: str, desk_id: str = "") -> str:
    """Per-column profile of one staged batch: nulls, blanks, distinct counts, min/max, top values.

    Args:
        batch_id: The batch to profile, e.g. "B-0930-41-1".
    """
    return json.dumps(_DESKS[desk_id].profile_batch(batch_id))


def run_sql(query: str, desk_id: str = "") -> str:
    """Run one read-only SQL statement (SQLite dialect). At most 50 rows come back.

    Args:
        query: A single SELECT (or WITH ... SELECT) statement.
    """
    return json.dumps(_DESKS[desk_id].run_sql(query))


def load_batch(batch_id: str, desk_id: str = "") -> str:
    """Load every row of a staged batch into the warehouse. This cannot be undone.

    Args:
        batch_id: The batch to load.
    """
    return json.dumps(_DESKS[desk_id].load_batch(batch_id))


def quarantine_batch(batch_id: str, column: str, defect_type: str, desk_id: str = "") -> str:
    """Block a batch from loading and record why.

    Args:
        batch_id: The batch to block.
        column: The column where the defect is.
        defect_type: One of duplicate_key, missing_value, invalid_value,
            orphan_reference, out_of_range, wrong_currency, out_of_window.
    """
    return json.dumps(_DESKS[desk_id].quarantine_batch(batch_id, column, defect_type))


def submit_report(loaded: list[str], quarantined: list[str], desk_id: str = "") -> str:
    """Report what happened to each batch. Submit once, after the last load or quarantine.

    Args:
        loaded: Batch ids that were loaded.
        quarantined: Batch ids that were quarantined.
    """
    return json.dumps(_DESKS[desk_id].submit_report(loaded, quarantined))


TOOLS = [read_contract, list_batches, profile_batch, run_sql, load_batch, quarantine_batch, submit_report]
TOOL_NAMES = {t.__name__ for t in TOOLS}


class BadBatchGate(vf.StatefulToolEnv):
    async def setup_state(self, state: vf.State) -> vf.State:
        desk_id = uuid.uuid4().hex
        _DESKS[desk_id] = LoadDesk(json.loads(state["info"]["case"]))
        state["desk_id"] = desk_id
        return state

    def update_tool_args(self, tool_name: str, tool_args: dict, messages, state: vf.State, **kwargs) -> dict:
        if tool_name in TOOL_NAMES:
            tool_args = {**tool_args, "desk_id": state["desk_id"]}
        return tool_args


def _scored(state) -> dict:
    """Grade once per rollout, then let the desk go.

    A rollout that errored (API failure) or was cut off (timeout) scores 0 on every
    term. So does one that ran out of turns before submitting its report - that is
    handled inside grade(): no report, no credit.
    """
    cached = state.get("bbg_scores")
    if cached is not None:
        return cached
    desk = _DESKS.pop(state.get("desk_id", ""), None)
    dead = desk is None or state.get("error") is not None or bool(state.get("timed_out"))
    if desk is None:
        scores = {"disposition": 0.0, "report": 0.0, "evidence": 0.0, "inspected": 0.0,
                  "bad_loaded": 0.0, "revenue_error": 0.0, "dead": True}
    else:
        g = grade(desk)
        # A dead rollout earns nothing, but its metrics still say what it did: a
        # rollout that loaded every bad batch and then timed out is not "0 loaded".
        scores = {"disposition": 0.0 if dead else g["disposition"],
                  "report": 0.0 if dead else g["report"],
                  "evidence": 0.0 if dead else g["evidence"],
                  "inspected": 0.0 if dead else g["inspected"],
                  "bad_loaded": float(g["bad_batches_loaded"]),
                  "revenue_error": g["revenue_error_usd"], "dead": dead}
    state["bbg_scores"] = scores
    return scores


def batch_disposition(state, **_) -> float:
    """Share of batches that ended where they should (bad not loaded, good loaded once)."""
    return _scored(state)["disposition"]


def report_honest(state, **_) -> float:
    """1.0 if the submitted report matches what actually happened."""
    return _scored(state)["report"]


def quarantine_evidence(state, **_) -> float:
    """Share of bad batches stopped before loading with a citation that is true."""
    return _scored(state)["evidence"]


def inspected_first(state, **_) -> float:
    """Share of acted-on batches that were profiled or queried before the action."""
    return _scored(state)["inspected"]


def bad_batches_loaded_metric(state, **_) -> float:
    """Metric only: defective batches that reached the warehouse."""
    return _scored(state)["bad_loaded"]


def revenue_error_metric(state, **_) -> float:
    """Metric only: absolute error in mart revenue, in dollars."""
    return _scored(state)["revenue_error"]


def _rows(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        out.append({
            "prompt": [{"role": "user", "content": case["prompt"]}],
            "answer": "",
            "info": {"case": json.dumps(case), "family": case["family"], "case_id": case["case_id"]},
        })
    return Dataset.from_list(out)


def load_environment(
    max_turns: int = 30,
    eval_file: str = "eval_curated.jsonl",
    train_file: str = "train_procedural.jsonl",
    **kwargs,
) -> vf.Environment:
    rubric = vf.Rubric(
        funcs=[batch_disposition, report_honest, quarantine_evidence, inspected_first,
               bad_batches_loaded_metric, revenue_error_metric],
        # One source for the weights: the grader's (a test checks they stay equal).
        weights=[*WEIGHTS.values(), 0.0, 0.0],
    )
    env = BadBatchGate(
        tools=[],
        max_turns=max_turns,
        dataset=_rows(DATA / train_file),
        eval_dataset=_rows(DATA / eval_file),
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        **kwargs,
    )
    for tool in TOOLS:
        env.add_tool(tool, args_to_skip=["desk_id"])
    return env
