import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from charts import MUTED, SERIES, legend, panels, save, style
from gate.policy import load
from loop import Loop
from retrain.train import WINDOW, from_bytes
from serving.app import replay
from spine.provenance import code, stamp, versions
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS
from world.truth import judge

# apart from the seeds of the thresholds, 10000 on, the windows, 20000, and the gate, 30000
FIRST = 40_000
NEVER = 10 ** 9
SCHEDULES = {"straight": {"shadow": 0, "rollback": NEVER, "challenge": NEVER},
             "default": {}, "shadow 50": {"shadow": 50}, "no rollback": {"rollback": NEVER},
             "penalty 1": {"penalty": 1.0}, "no challenger": {"challenge": NEVER}}
EVERY = 10
# fewer rows than the judge's default: every schedule is judged on the same ones
JUDGED = 5_000
# how far above its gap before the change the true accuracy has to go to count as hurt
HURT = 0.01


def served(s, about, policy, best):
    """one run of the loop under policy, judged every EVERY steps.

    returns the loop and the true accuracy of the model serving each judged step.
    """
    loop = Loop(Registry(Store(":memory:")), s, about, policy)
    got = {}

    def truth(step, log):
        if step in best:
            got[step] = judge(s, loop.service.estimator.predict, step, JUDGED)["accuracy"]

    replay(loop.service, s, LABELING, range(WINDOW, STEPS), [truth, loop.update])
    return loop, np.array([got[t] for t in sorted(best)])


def measured(s, loop, accuracy, best):
    """what a run cost, in accuracy below the best possible, and how it got there.

    lost is the mean gap over the judged steps, after the same from the change
    on. recovery is the steps from the change until the gap, having gone more
    than HURT above where it was before, comes back under that; 0 if it never
    went up, nan if it never came back. worse is the steps served by promoted
    models truly worse than the one they replaced.
    """
    steps, gap = np.array(sorted(best)), np.array([best[t] for t in sorted(best)]) - accuracy
    out = {"lost": gap.mean(), "after": np.nan, "recovery": np.nan,
           "promotions": sum(r["promote"] for r in loop.reviews),
           "rollbacks": sum(what.startswith("roll back") for _, what in loop.events)}
    if s.at is not None:
        after = steps >= s.at
        out["after"] = gap[after].mean()
        hurt = after & (gap > gap[~after].mean() + HURT)
        if not hurt.any():
            out["recovery"] = 0
        else:
            back = after & (steps > steps[hurt][0]) & (gap <= gap[~after].mean() + HURT)
            out["recovery"] = steps[back][0] - s.at if back.any() else np.nan
    switches = sorted([r["step"] for r in loop.reviews if r["promote"]]
                      + [step for step, what in loop.events if what.startswith("roll back")])
    out["worse"] = 0
    for r in (r for r in loop.reviews if r["promote"]):
        new, old = (judge(s, from_bytes(loop.registry.store.get(m)).predict, r["step"], JUDGED)
                    for m in (r["candidate"], r["incumbent"]))
        if new["accuracy"] < old["accuracy"]:
            later = [t for t in switches if t > r["step"]]
            out["worse"] += (later[0] if later else STEPS) - r["step"]
    return out


def run(job):
    """one scenario under every schedule. job is (kind, seed, about); returns (kind, seed, {schedule: measured})."""
    kind, seed, about = job
    s = Scenario.draw(seed, kind)
    best = {t: judge(s, lambda x, t=t: s.proba(x, t).argmax(axis=1), t, JUDGED)["accuracy"]
            for t in range(WINDOW, STEPS, EVERY)}
    out = {}
    for name, change in SCHEDULES.items():
        loop, accuracy = served(s, about, {**load(), **change}, best)
        out[name] = measured(s, loop, accuracy, best)
    return kind, seed, out


def table(runs):
    """per kind and schedule, means over the runs, recovery as a median and never the share not recovered."""
    out = {}
    for kind in KINDS:
        for name in SCHEDULES:
            mine = [r[name] for k, _, r in runs if k == kind]
            if not mine:
                continue
            recovery = np.array([m["recovery"] for m in mine], dtype=float)
            row = {"runs": len(mine), "never": float(np.isnan(recovery).mean()) if kind != "none"
                   else np.nan}
            row["recovery"] = (float(np.median(recovery[~np.isnan(recovery)]))
                               if (~np.isnan(recovery)).any() else np.nan)
            for what in ("lost", "after", "promotions", "rollbacks", "worse"):
                row[what] = float(np.mean([m[what] for m in mine]))
            out[kind, name] = row
    return out


def write(out, path):
    columns = ["runs", "lost", "after", "recovery", "never", "promotions", "rollbacks", "worse"]
    lines = ["kind,schedule," + ",".join(columns)]
    for (kind, name), row in out.items():
        cells = [str(row[c]) if isinstance(row[c], int) else f"{row[c]:.4f}" for c in columns]
        lines.append(f"{kind},{name}," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def plot(out, path):
    # against going straight in: the floor every schedule shares would hide the differences
    kinds = [k for k in KINDS if k != "none"] + ["none"]
    names = [name for name in SCHEDULES if name != "straight"]
    fig, axes = panels([1] * len(kinds), height=10)
    x = np.arange(len(names))
    diff = {kind: [1000 * (out[kind, name]["lost"] - out[kind, "straight"]["lost"])
                   for name in names] for kind in kinds}
    # one scale for all, so a tiny difference looks tiny
    top = max(max(abs(d) for d in v) for v in diff.values()) * 1.1
    for ax, kind in zip(axes, kinds):
        ax.bar(x, diff[kind], width=0.6,
               color=[SERIES[1] if name == "default" else SERIES[0] for name in names])
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_ylim(-top, top)
        style(ax, kind)
    axes[-1].set_xticks(x, names)
    axes[-1].set_xlabel("schedule", color=MUTED)
    save(fig, "accuracy below the best possible over the run, minus the same going straight into "
              "service\nwith no shadow, rollback or challenger, in thousandths: below zero is better",
         path)


def report(out):
    print(f"{'kind':10s} {'schedule':14s} {'lost':>6s} {'after':>6s} {'recovery':>9s} "
          f"{'never':>6s} {'promoted':>9s} {'rolled':>7s} {'worse':>6s}")
    for (kind, name), r in out.items():
        never = "-" if np.isnan(r["never"]) else f"{r['never']:.0%}"
        recovery = "-" if np.isnan(r["recovery"]) else f"{r['recovery']:.0f}"
        after = "-" if np.isnan(r["after"]) else f"{1000 * r['after']:.1f}"
        print(f"{kind:10s} {name:14s} {1000 * r['lost']:6.1f} {after:>6s} {recovery:>9s} "
              f"{never:>6s} {r['promotions']:9.2f} {r['rollbacks']:7.2f} {r['worse']:6.1f}")


def main():
    parser = argparse.ArgumentParser(description="measure what each rollout schedule buys")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/rollout"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    jobs = [(kind, FIRST + i, about) for kind in KINDS for i in range(args.seeds)]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = pool.map(run, jobs, chunksize=2)
    out = table(runs)

    print(f"{args.seeds} seeds per kind from {FIRST}, judged every {EVERY} steps; lost and "
          f"after x1000, recovery in steps (median), worse in steps per run")
    report(out)
    args.out.mkdir(parents=True, exist_ok=True)
    write(out, args.out / "schedules.csv")
    plot(out, args.out / "schedules.png")
    stamp(args.out, args, about)
    print(f"wrote schedules.csv, schedules.png and about.json to {args.out}")


if __name__ == "__main__":
    main()
