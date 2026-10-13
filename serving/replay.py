import argparse
from pathlib import Path

import numpy as np

from charts import MUTED, SERIES, mark, panel, panels, rolling, save
from retrain.train import WINDOW, train
from serving.app import Service, replay
from serving.log import UNKNOWN
from spine.provenance import code, versions
from spine.registry import fresh
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS, describe

RECENT = 50
SMOOTH = 25


def lag(s, log, first, last):
    """for each served step t from first to last, as arrays indexed by t - first.

    true is the expected accuracy of what was served at t, under the true
    P(y|x) at t. known is the accuracy over the RECENT steps up to t, from the
    labels that had come back by t, and labels how many it rests on (known is
    nan when there are none).
    """
    log = log.as_of(last)
    pred = log["proba"].argmax(axis=1)
    hit = pred == log["label"]
    n = last - first + 1
    true, known, labels = np.empty(n), np.full(n, np.nan), np.zeros(n, dtype=np.int64)
    for i, t in enumerate(range(first, last + 1)):
        lo, hi = np.searchsorted(log["step"], [t, t + 1])
        p = s.proba(log["x"][lo:hi], t)
        true[i] = p[np.arange(hi - lo), pred[lo:hi]].mean()
        lo = np.searchsorted(log["step"], t - RECENT + 1)
        back = (log["arrived"][lo:hi] != UNKNOWN) & (log["arrived"][lo:hi] <= t)
        labels[i] = back.sum()
        if labels[i]:
            known[i] = hit[lo:hi][back].mean()
    return {"true": true, "known": known, "labels": labels}


def halfway(s, r, first):
    """the step where known got halfway from the true accuracy before the change to after it.

    None when there's no change, when the true accuracy didn't drop 2 points,
    or when known never got that low.
    """
    if s.at is None:
        return None
    at = s.at - first
    before, after = r["true"][:at].mean(), r["true"][at + RECENT:].mean()
    if before - after < 0.02:
        return None
    down = np.flatnonzero(r["known"][at:] <= (before + after) / 2)
    return first + at + int(down[0]) if len(down) else None


def plot(s, r, first, path):
    fig, (top, bottom) = panels([3, 2])
    x = np.arange(first + RECENT - 1, first + len(r["true"]))
    lines = [("true", rolling(r["true"], SMOOTH)[RECENT - SMOOTH:], SERIES[0], 1.6),
             ("known at the step", r["known"][RECENT - 1:], SERIES[1], 1.6)]
    panel(top, x, lines, "accuracy")
    panel(bottom, x, [("labels", r["labels"][RECENT - 1:], MUTED, 1.6)],
          f"labels back, last {RECENT} steps")
    bottom.set_xlabel("step", color=MUTED)
    if s.at is not None:
        mark((top, bottom), s.at, f"change at {s.at}")
    save(fig, describe(s) + f"\nserved by the model trained at step {first}; true is a mean "
              f"over {SMOOTH} steps, known uses the labels back by each step", path)


def main():
    parser = argparse.ArgumentParser(description="replay a scenario to see what's known when")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default="concept")
    parser.add_argument("--out", type=Path, default=Path("results/replay"))
    args = parser.parse_args()

    s = Scenario.draw(args.seed, args.kind)
    registry = fresh(f"replay-{s.kind}-{s.seed}")
    model = train(registry, s, WINDOW, None, {"code": code(), "versions": versions()}, LABELING)
    service = Service(registry, model)
    replay(service, s, LABELING, range(WINDOW, STEPS))
    service.log.save(registry.store.db)
    r = lag(s, service.log, WINDOW, STEPS - 1)

    print(describe(s))
    print(f"served steps {WINDOW}-{STEPS - 1} with {model[:12]}, trained at step {WINDOW}")
    if s.at is not None:
        at = s.at - WINDOW
        for name, part in [("before", slice(0, at)), ("after", slice(at + RECENT, None))]:
            print(f"{name:7s} true {r['true'][part].mean():.3f}, known "
                  f"{np.nanmean(r['known'][part]):.3f}, labels behind it {r['labels'][part].mean():.0f}")
    step = halfway(s, r, WINDOW)
    if step is not None:
        print(f"what was known got halfway down at step {step}, {step - s.at} steps after the change")
    else:
        print("the true accuracy didn't drop 2 points")
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.out / f"{s.kind}-{s.seed}"
    steps = np.arange(WINDOW, STEPS)
    np.savetxt(name.with_suffix(".csv"), np.column_stack([steps, r["true"], r["known"], r["labels"]]),
               delimiter=",", header="step,true,known,labels", comments="",
               fmt=["%d", "%.6f", "%.6f", "%d"])
    plot(s, r, WINDOW, name.with_suffix(".png"))
    print(f"wrote {name}.csv and {name}.png, the log is in {registry.store.path}")


if __name__ == "__main__":
    main()
