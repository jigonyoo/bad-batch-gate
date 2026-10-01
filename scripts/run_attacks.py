"""Every attacker in attackers.py against the eval split. No API key."""
import json
import pathlib
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
DATA = pathlib.Path(__file__).resolve().parents[1] / "bad_batch_gate" / "data"

from bad_batch_gate.agents import careful_agent  # noqa: E402
from bad_batch_gate.attackers import ATTACKERS  # noqa: E402
from bad_batch_gate.desk import LoadDesk  # noqa: E402
from bad_batch_gate.grader import grade  # noqa: E402

if __name__ == "__main__":
    cases = [json.loads(l) for l in open(DATA / "eval_curated.jsonl", encoding="utf-8")]
    print(f"{'agent':22}{'reward':>9}{'disp':>8}{'report':>8}{'evid':>8}{'insp':>8}{'bad loaded':>12}{'revenue err $':>16}")
    for name, agent in {"careful (target)": careful_agent, **ATTACKERS}.items():
        gs = []
        for case in cases:
            desk = LoadDesk(case)
            agent(desk)
            gs.append(grade(desk))
        # Exact mean at 9 decimals, so the digits do not depend on the Python version.
        m = lambda k: float(sum(Fraction(f"{g[k]:.9f}") for g in gs) / len(gs))  # noqa: E731
        print(f"{name:22}{m('reward'):>9.4f}{m('disposition'):>8.4f}{m('report'):>8.4f}{m('evidence'):>8.4f}"
              f"{m('inspected'):>8.4f}{sum(g['bad_batches_loaded'] for g in gs):>12}"
              f"{sum(g['revenue_error_usd'] for g in gs):>16,.2f}")
