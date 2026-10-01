"""bad-batch-gate.

`load_environment` is imported lazily on purpose. The world, the grader, the
dataset and the reference agents run on the standard library alone, and
`scripts/run_report.py` and `scripts/run_attacks.py` are meant to stay that
way - importing `verifiers` eagerly here would make every offline check
require the full RL stack.
"""

__all__ = ["load_environment"]


def __getattr__(name: str):
    if name == "load_environment":
        try:
            from .environment import load_environment
        except AttributeError as exc:
            # An AttributeError raised here would be reported by Python as "cannot
            # import name 'load_environment'", hiding the real cause - usually a
            # verifiers build without the StatefulToolEnv this package targets.
            raise ImportError(f"bad_batch_gate could not load its environment: {exc!r}. "
                              "It needs verifiers>=0.3.1,<0.3.2.dev0.") from exc

        return load_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
