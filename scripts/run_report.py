"""Runs the three reference agents over the eval split and prints the tables in
the README: baselines, per family, and an ablation. No API key, no network,
standard library only."""
import json
import pathlib
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
DATA = pathlib.Path(__file__).resolve().parents[1] / "bad_batch_gate" / "data"

from bad_batch_gate.agents import careful_agent, naive_agent, refuse_all_agent  # noqa: E402
from bad_batch_gate.desk import LoadDesk  # noqa: E402
from bad_batch_gate.grader import WEIGHTS, correct_revenue, grade  # noqa: E402

AGENTS = {"naive": naive_agent, "careful": careful_agent, "refuse-all": refuse_all_agent}


def run(agent, cases):
    out = []
    for case in cases:
        desk = LoadDesk(case)
        agent(desk)
        out.append((case, grade(desk)))
    return out


def mean(xs):
    """Exact mean, so the printed digits do not depend on the Python version.
    (3.12 changed how sum() adds floats; 0.3875 printed as 0.387 on 3.11 and
    0.388 on 3.12 before this.) Each value is taken at 9 decimals."""
    xs = [Fraction(f"{x:.9f}") for x in xs]
    return float(sum(xs) / len(xs))


if __name__ == "__main__":
    cases = [json.loads(l) for l in open(DATA / "eval_curated.jsonl", encoding="utf-8")]
    n_batches = sum(len(c["batches"]) for c in cases)
    n_bad = sum(b["truth"]["defective"] for c in cases for b in c["batches"])
    print(f"eval split: {len(cases)} cases, {n_batches} batches, {n_bad} defective\n")
    results = {name: run(a, cases) for name, a in AGENTS.items()}

    print(f"{'':22}" + "".join(f"{n:>14}" for n in AGENTS))
    rows = [
        ("reward (mean)", lambda r: f"{mean(g['reward'] for _, g in r):.4f}"),
        ("disposition (mean)", lambda r: f"{mean(g['disposition'] for _, g in r):.4f}"),
        ("report matches", lambda r: f"{sum(g['report'] == 1.0 for _, g in r)}/{len(r)}"),
        ("defective loaded", lambda r: f"{sum(g['bad_batches_loaded'] for _, g in r)}/{n_bad}"),
        ("revenue error $", lambda r: f"{sum(g['revenue_error_usd'] for _, g in r):,.2f}"),
        ("ledgers intact", lambda r: f"{sum(g['ledger_intact'] for _, g in r)}/{len(r)}"),
    ]
    for label, f in rows:
        print(f"{label:22}" + "".join(f"{f(results[n]):>14}" for n in AGENTS))

    print("\nper family (mean reward):")
    fams = list(dict.fromkeys(c["family"] for c in cases))
    print(f"{'':22}" + "".join(f"{n:>14}" for n in AGENTS))
    for fam in fams:
        print(f"{fam:22}" + "".join(
            f"{mean(g['reward'] for c, g in results[n] if c['family'] == fam):>14.4f}" for n in AGENTS))

    print("\nablation (mean reward with one term removed, weights not renormalised):")
    print(f"{'':22}" + "".join(f"{n:>14}" for n in AGENTS))
    for drop in [None, *WEIGHTS]:
        label = "full" if drop is None else f"no {drop}"
        vals = []
        for n in AGENTS:
            vals.append(mean(sum(WEIGHTS[k] * g[k] for k in WEIGHTS if k != drop) for _, g in results[n]))
        print(f"{label:22}" + "".join(f"{v:>14.4f}" for v in vals))

    print("\nlargest naive revenue errors (mart after naive vs mart if handled correctly):")
    worst = sorted(results["naive"], key=lambda cg: -cg[1]["revenue_error_usd"])[:3]
    for c, _ in worst:
        desk = LoadDesk(c)
        naive_agent(desk)
        print(f"  {c['case_id']:26} naive ${desk.warehouse_revenue():>12,.2f}   correct ${correct_revenue(c):>12,.2f}")
