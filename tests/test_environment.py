"""verifiers wiring. Skipped when verifiers is not installed; the offline tests
above do not need it."""
import asyncio
import importlib
import json
import sys

import pytest

vf = pytest.importorskip("verifiers")


def test_offline_modules_import_without_verifiers(monkeypatch):
    """The grader and the agents must not pull in the RL stack."""
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name == "verifiers" or name.startswith("verifiers."):
                raise ImportError("blocked for this test")
            return None
    for mod in [m for m in sys.modules if m.startswith("bad_batch_gate")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(sys, "meta_path", [Block(), *sys.meta_path])
    importlib.import_module("bad_batch_gate.grader")
    importlib.import_module("bad_batch_gate.agents")
    importlib.import_module("bad_batch_gate.attackers")


def test_load_environment_builds_with_seven_tools():
    from bad_batch_gate import load_environment
    env = load_environment()
    names = {t.__name__ for t in env.tools}
    assert names == {"read_contract", "list_batches", "profile_batch", "run_sql",
                     "load_batch", "quarantine_batch", "submit_report"}
    assert len(env.eval_dataset) == 32


def test_desk_id_is_hidden_from_the_model_schema():
    from bad_batch_gate import load_environment
    env = load_environment()
    defs = env.tool_defs
    assert defs, "no tool definitions to inspect"   # a vacuous pass would hide a leak
    schemas = json.dumps(defs, default=str)
    assert "run_sql" in schemas and "quarantine_batch" in schemas
    assert "desk_id" not in schemas


@pytest.mark.parametrize("flag", [{"error": "HTTP 402"}, {"timed_out": True}])
def test_errored_or_timed_out_rollout_scores_zero_even_with_a_live_desk(flag):
    """The desk exists and a report was filed, but the rollout died: no credit."""
    from bad_batch_gate import environment as E
    from bad_batch_gate.agents import careful_agent
    from bad_batch_gate.desk import LoadDesk
    case = json.loads(E._rows(E.DATA / "eval_curated.jsonl")[0]["info"]["case"])
    E._DESKS["dead"] = LoadDesk(case)
    careful_agent(E._DESKS["dead"])
    state = {"desk_id": "dead", **flag}
    assert E.batch_disposition(state) == 0.0
    assert E.report_honest(state) == 0.0
    assert E.quarantine_evidence(state) == 0.0
    assert E.inspected_first(state) == 0.0
    assert "dead" not in E._DESKS


def test_run_out_of_turns_without_a_report_scores_zero():
    from bad_batch_gate import environment as E
    from bad_batch_gate.desk import LoadDesk
    case = json.loads(E._rows(E.DATA / "eval_curated.jsonl")[0]["info"]["case"])
    E._DESKS["noreport"] = LoadDesk(case)
    state = {"desk_id": "noreport"}
    assert E.batch_disposition(state) + E.report_honest(state) + E.quarantine_evidence(state) == 0.0


def test_sql_bytes_come_back_as_json():
    from bad_batch_gate import environment as E
    from bad_batch_gate.desk import LoadDesk
    case = json.loads(E._rows(E.DATA / "eval_curated.jsonl")[0]["info"]["case"])
    E._DESKS["b"] = LoadDesk(case)
    out = json.loads(E.run_sql("SELECT X'DEADBEEF' AS b", desk_id="b"))
    assert out["rows"][0][0].startswith("0x")
    E._DESKS.pop("b")


def test_tools_drive_the_same_desk_the_grader_reads():
    from bad_batch_gate import environment as E
    from bad_batch_gate.desk import LoadDesk
    case = json.loads(E._rows(E.DATA / "eval_curated.jsonl")[0]["info"]["case"])
    E._DESKS["t"] = LoadDesk(case)
    batches = json.loads(E.list_batches(desk_id="t"))
    for b in batches:
        E.profile_batch(b["batch_id"], desk_id="t")
        E.load_batch(b["batch_id"], desk_id="t")
    E.submit_report([b["batch_id"] for b in batches], [], desk_id="t")
    state = {"desk_id": "t"}
    assert E.report_honest(state) == 1.0
    assert E.inspected_first(state) == 1.0
    assert "t" not in E._DESKS


def test_dead_rollout_keeps_its_metrics():
    """Scores go to 0, but bad_batches_loaded still reports what happened."""
    from bad_batch_gate import environment as E
    from bad_batch_gate.agents import naive_agent
    from bad_batch_gate.desk import LoadDesk
    rows = E._rows(E.DATA / "eval_curated.jsonl")
    case = next(json.loads(r["info"]["case"]) for r in rows if json.loads(r["info"]["case"])["family"] == "blank-amount")
    E._DESKS["late"] = LoadDesk(case)
    naive_agent(E._DESKS["late"])
    state = {"desk_id": "late", "timed_out": True}
    assert E.batch_disposition(state) == 0.0
    assert E.bad_batches_loaded_metric(state) == 1.0


def test_rubric_uses_the_graders_weights():
    from bad_batch_gate import load_environment
    from bad_batch_gate.grader import WEIGHTS
    env = load_environment()
    ours = [r for r in getattr(env.rubric, "rubrics", [env.rubric])
            if any(f.__name__ == "batch_disposition" for f in r.funcs)]
    assert len(ours) == 1
    assert list(ours[0].weights) == [*WEIGHTS.values(), 0.0, 0.0]
    # Every other rubric verifiers adds (turn and tool counters) must weigh nothing.
    others = [w for r in getattr(env.rubric, "rubrics", []) if r is not ours[0] for w in r.weights]
    assert all(w == 0.0 for w in others)


def _scripted_client(forge_desk_id=False):
    """A client that answers from a script instead of a model: no network, no key.
    It lists the batches, profiles each, loads customers first, and reports."""
    import time
    from verifiers.legacy.clients.client import Client
    from verifiers.legacy.types import Response, ResponseMessage, ToolCall

    def get(m, k):
        return m.get(k) if isinstance(m, dict) else getattr(m, k, None)

    class Scripted(Client):
        def setup_client(self, config):
            return None

        async def to_native_tool(self, tool):
            return tool

        async def to_native_prompt(self, messages):
            return messages, {}

        async def raise_from_native_response(self, r):
            return None

        async def from_native_response(self, r):
            return r

        async def close(self):
            return None

        async def get_native_response(self, prompt, model, sampling_args, tools=None, **kw):
            n = sum(1 for m in prompt if get(m, "role") == "assistant")
            extra = {"desk_id": "forged"} if forge_desk_id else {}
            calls = []
            if n == 0:
                calls = [("list_batches", extra)]
            elif n == 1:
                tool_msgs = [m for m in prompt if get(m, "role") == "tool"]
                self.batches = json.loads(get(tool_msgs[-1], "content"))
                calls = [("profile_batch", {"batch_id": b["batch_id"], **extra}) for b in self.batches]
            elif n == 2:
                order = sorted(self.batches, key=lambda b: b["table"] != "customers")
                calls = [("load_batch", {"batch_id": b["batch_id"], **extra}) for b in order]
            elif n == 3:
                calls = [("submit_report", {"loaded": [b["batch_id"] for b in self.batches],
                                            "quarantined": [], **extra})]
            msg = ResponseMessage(
                content=None if calls else "done.",
                tool_calls=[ToolCall(id=f"c{n}_{i}", name=a, arguments=json.dumps(b))
                            for i, (a, b) in enumerate(calls)] or None,
                finish_reason="tool_calls" if calls else "stop", is_truncated=False)
            return Response(id=f"r{n}", created=int(time.time()), model=model, message=msg)

    return Scripted(object())


@pytest.mark.parametrize("forge", [False, True])
def test_a_scripted_rollout_through_the_real_env_scores_like_the_grader(forge):
    """End to end through StatefulToolEnv: setup_state creates the desk,
    update_tool_args injects its id (overriding anything the model sends),
    the rubric scores it, and the desk is released. Compared with grade() on the
    same actions run directly."""
    from bad_batch_gate import environment as E
    from bad_batch_gate import load_environment
    from bad_batch_gate.desk import LoadDesk
    from bad_batch_gate.grader import grade
    env = load_environment()
    ds = env.eval_dataset
    # Cases from different families, so a desk built from the wrong case shows up
    # in the revenue metric even when the reward happens to match.
    fams, picks = set(), []
    for i in range(len(ds)):
        f = json.loads(ds[i]["info"]["case"])["family"]
        if f not in fams and f not in ("clean-control", "legit-edge"):
            fams.add(f)
            picks.append(i)
    for i in picks[:4]:
        row = ds[i]
        inp = {"prompt": row["prompt"], "example_id": i, "answer": "", "info": row["info"]}
        out = asyncio.run(env.run_rollout(inp, _scripted_client(forge), "scripted", {}))
        case = json.loads(row["info"]["case"])
        d = LoadDesk(case)
        bs = d.list_batches()
        for b in bs:
            d.profile_batch(b["batch_id"])
        for b in sorted(bs, key=lambda b: b["table"] != "customers"):
            d.load_batch(b["batch_id"])
        d.submit_report([b["batch_id"] for b in bs], [])
        g = grade(d)
        assert out["reward"] == pytest.approx(g["reward"]), case["case_id"]
        assert out["metrics"]["revenue_error_metric"] == pytest.approx(g["revenue_error_usd"]), case["case_id"]
        assert out["metrics"]["bad_batches_loaded_metric"] == g["bad_batches_loaded"] == 1
        assert out["reward"] > 0
    assert not E._DESKS


def test_default_turn_limit_is_the_documented_one():
    from bad_batch_gate import load_environment
    assert load_environment().max_turns == 30


def test_a_verifiers_without_statefultoolenv_is_reported_as_such(monkeypatch):
    """Without the re-raise in __init__, Python would say only "cannot import name
    'load_environment'" and hide which verifiers build is the problem."""
    import bad_batch_gate
    real = vars(vf).get("__getattr__")

    def missing(name):                     # verifiers resolves its names lazily
        if name == "StatefulToolEnv":
            raise AttributeError(f"module 'verifiers' has no attribute {name!r}")
        return real(name)
    monkeypatch.setattr(vf, "__getattr__", missing)
    monkeypatch.delitem(sys.modules, "bad_batch_gate.environment", raising=False)
    with pytest.raises(ImportError, match=r"verifiers>=0\.3\.1"):
        bad_batch_gate.load_environment


def test_load_environment_takes_its_documented_arguments():
    from bad_batch_gate import load_environment
    env = load_environment(max_turns=40, eval_file="train_procedural.jsonl")
    assert env.max_turns == 40
    assert len(env.eval_dataset) == 320


def test_the_model_sees_only_the_case_prompt_and_the_system_prompt():
    from bad_batch_gate import environment as E
    rows = E._rows(E.DATA / "eval_curated.jsonl")
    for r in rows:
        case = json.loads(r["info"]["case"])
        assert r["prompt"] == [{"role": "user", "content": case["prompt"]}]
    for word in ("defective", "truth", "accepted", "family", "case_id"):
        assert word not in E.SYSTEM_PROMPT


def test_concurrent_rollouts_each_get_their_own_desk():
    """Rollouts run concurrently in one process; each must act on its own case."""
    from bad_batch_gate import environment as E
    from bad_batch_gate import load_environment
    from bad_batch_gate.desk import LoadDesk
    from bad_batch_gate.grader import grade
    env = load_environment()
    ds = env.eval_dataset
    fams, picks = set(), []
    for i in range(len(ds)):
        f = json.loads(ds[i]["info"]["case"])["family"]
        if f not in fams and f not in ("clean-control", "legit-edge"):
            fams.add(f)
            picks.append(i)
    picks = picks[:4]

    async def many():
        inputs = [{"prompt": ds[i]["prompt"], "example_id": i, "answer": "", "info": ds[i]["info"]} for i in picks]
        return await asyncio.gather(*(env.run_rollout(inp, _scripted_client(), "scripted", {}) for inp in inputs))

    outs = asyncio.run(many())
    for i, out in zip(picks, outs):
        case = json.loads(ds[i]["info"]["case"])
        d = LoadDesk(case)
        bs = d.list_batches()
        for b in bs:
            d.profile_batch(b["batch_id"])
        for b in sorted(bs, key=lambda b: b["table"] != "customers"):
            d.load_batch(b["batch_id"])
        d.submit_report([b["batch_id"] for b in bs], [])
        assert out["metrics"]["revenue_error_metric"] == pytest.approx(grade(d)["revenue_error_usd"]), case["case_id"]
        assert out["reward"] == pytest.approx(grade(d)["reward"]), case["case_id"]
    assert not E._DESKS
