import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from eval.oracle import best_available
from gate.policy import load, naive, review
from loop import Loop
from retrain.train import WINDOW, from_bytes
from retrain.window import Window
from serving.app import replay
from spine.provenance import code, stamp, versions
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS
from world.truth import judge

# apart from every seed tuned on so far: thresholds 10000, windows 20000, gate 30000, rollout 40000
FIRST = 50_000
NEVER = 10 ** 9
EVERY = 10
JUDGED = 5_000
# a true difference smaller than the cost of a switch is no difference
GENUINE = load()["switch"]
# how far the true accuracy has to fall for an alarm to have something behind it
HURT = 0.01
# each changes one thing from the defaults, except the naive pipeline, which changes all of them
POLICIES = {
    "naive": {"trigger": "naive", "gate": naive,
              "policy": {"shadow": 0, "rollback": NEVER, "challenge": NEVER}},
    "default": {},
    "penalty 1": {"policy": {"penalty": 1.0}},
    "whole window": {"window": Window(old_share=1.0)},
    "shadow 50": {"policy": {"shadow": 50}},
}


def looped(s, about, setup, steps):
    """one run of the loop as setup says, and (step, serving model, true accuracy) every judged step."""
    loop = Loop(Registry(Store(":memory:")), s, about, {**load(), **setup.get("policy", {})},
                window=setup.get("window", Window()), trigger=setup.get("trigger", "informed"),
                gate=setup.get("gate", review))
    timeline = []

    def truth(step, log):
        if step in steps:
            accuracy = judge(s, loop.service.estimator.predict, step, JUDGED)["accuracy"]
            timeline.append((step, loop.service.model, accuracy))

    replay(loop.service, s, LABELING, range(WINDOW, STEPS), [truth, loop.update])
    return loop, timeline


def gains(s, loop):
    """the true gain of every candidate the gate looked at over the incumbent, at that step."""
    def accuracy(model, step):
        return judge(s, from_bytes(loop.registry.store.get(model)).predict, step, JUDGED)["accuracy"]
    return [accuracy(r["candidate"], r["step"]) - accuracy(r["incumbent"], r["step"])
            for r in loop.reviews]


def alarms(s, loop, timeline):
    """(detection delay, decisions, false ones) of the trigger.

    the delay is the steps from the change to the first decision after it, nan
    without one. a decision is false when the model in service had not lost
    HURT of the true accuracy it started serving with.
    """
    started, false = {}, 0
    for _, model, accuracy in timeline:
        started.setdefault(model, accuracy)
    for d in loop.decisions:
        _, model, accuracy = max((t for t in timeline if t[0] <= d["step"]), key=lambda t: t[0])
        false += accuracy > started[model] - HURT
    after = [d["step"] - s.at for d in loop.decisions if s.at is not None and d["step"] >= s.at]
    return (int(after[0]) if after else np.nan), len(loop.decisions), int(false)


def recovery(s, steps, gap):
    """steps from the change until the gap to the best possible comes back near where it was.

    the gap has to go HURT above its mean before the change, then come back
    under that. 0 if it never went up, nan if it never came back or there was
    no change.
    """
    if s.at is None:
        return np.nan
    after = steps >= s.at
    level = gap[~after].mean() + HURT
    hurt = after & (gap > level)
    if not hurt.any():
        return 0
    back = after & (steps > steps[hurt][0]) & (gap <= level)
    return int(steps[back][0] - s.at) if back.any() else np.nan


def run(job):
    """one scenario under every policy. job is (kind, seed, about); returns a record per policy.

    regret is the accuracy times steps below the best model any of the
    policies had trained by each step, the same oracle for all of them.
    """
    kind, seed, about = job
    s = Scenario.draw(seed, kind)
    steps = np.arange(WINDOW, STEPS, EVERY)
    possible = np.array([judge(s, lambda x, t=t: s.proba(x, t).argmax(axis=1), t, JUDGED)["accuracy"]
                         for t in steps])
    runs = {name: looped(s, about, setup, set(steps.tolist())) for name, setup in POLICIES.items()}
    best, _ = best_available([loop.registry for loop, _ in runs.values()], steps,
                             lambda predict, t: judge(s, predict, t, JUDGED)["accuracy"])
    out = []
    for name, (loop, timeline) in runs.items():
        served = np.array([accuracy for _, _, accuracy in timeline])
        g = np.array(gains(s, loop))
        promoted = np.array([r["promote"] for r in loop.reviews], dtype=bool)
        delay, decisions, false = alarms(s, loop, timeline)
        out.append({
            "kind": kind, "seed": seed, "policy": name, "at": s.at,
            "regret": float(EVERY * (best - served).sum()),
            "recovery": recovery(s, steps, possible - served), "delay": delay,
            "decisions": decisions, "false alarms": false,
            "candidates": len(g), "wasted": int((g <= GENUINE).sum()),
            "promotions": int(promoted.sum()), "worse promoted": int((promoted & (g < -GENUINE)).sum()),
            "kept": int((~promoted).sum()), "better kept": int((~promoted & (g > GENUINE)).sum()),
            "rollbacks": sum(what.startswith("roll back") for _, what in loop.events),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="run the pipeline under several policies")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/runs"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    jobs = [(kind, FIRST + i, about) for kind in KINDS for i in range(args.seeds)]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = [r for rs in pool.map(run, jobs, chunksize=1) for r in rs]
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    stamp(args.out, args, about)
    print(f"{len(runs)} runs, {args.seeds} seeds per kind from {FIRST} under {len(POLICIES)} "
          f"policies, written to {args.out / 'runs.jsonl'}")


if __name__ == "__main__":
    main()
