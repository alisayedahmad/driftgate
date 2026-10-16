import tomllib
from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from pathlib import Path

from gate.compare import compare
from gate.segments import given_up

POLICY = Path(__file__).with_name("policy.toml")


def load(path=POLICY):
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class Verdict:
    promote: bool
    reason: str


def expected(gain, se):
    """(expected gain, expected loss) of promoting, both >= 0, if the true gain is N(gain, se²).

    their difference is gain; with se 0 one of them is.
    """
    if not se > 0:
        return max(gain, 0.0), max(-gain, 0.0)
    z = gain / se
    up = gain * 0.5 * (1 + erf(z / sqrt(2))) + se * exp(-z * z / 2) / sqrt(2 * pi)
    return up, up - gain


def decide(result, policy):
    """promote or hold, from compare()'s result and a policy as load() reads it.

    holds below policy["min_rows"] labeled rows, and when the candidate gives
    up a class, see gate.segments. otherwise promotes when the expected gain
    beats penalty times the expected loss, both taken over what the true gain
    could be given the rows: a likely small loss can outweigh a likely small
    gain, and with little evidence both are wide.
    """
    if result["rows"] < policy["min_rows"]:
        return Verdict(False, f"{result['rows']} labeled rows, {policy['min_rows']} needed")
    lost = given_up(result, policy)
    if lost:
        c, change, rows = lost[0]
        return Verdict(False, f"class {c} loses {-change:.3f} of recall on {rows} rows")
    up, down = expected(result["gain"], result["se"])
    weighed = f"expected gain {up:.4f} against {policy['penalty']:g} x {down:.4f} expected loss"
    return Verdict(up > policy["penalty"] * down, weighed)


def review(registry, candidate, incumbent, policy):
    """the gate on a candidate after it shadowed the incumbent policy["shadow"] steps.

    returns the step it looked at, the document naming the rows it compared on,
    compare()'s result and the Verdict. call it at that step or later.
    """
    look = registry.get(candidate)["step"] + policy["shadow"]
    doc, result = compare(registry, candidate, incumbent, look)
    return look, doc, result, decide(result, policy)
