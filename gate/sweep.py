import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from charts import MUTED, SERIES, legend, panels, save, style
from gate.compare import compare
from gate.policy import decide, load
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

# kept apart from the thresholds' seeds, 10000 on, and the windows', 20000 on
FIRST = 30_000
SHADOWS = (0, 10, 50, 100)
NEVER = 10 ** 9


def gates(policy):
    """each gate, as a function of (registry, candidate, incumbent, compare()'s result) to promote.

    paired, penalty and guard leave out the cost of a switch, default is the
    policy the loop runs with.
    """
    def paired(**change):
        return lambda registry, new, old, got: decide(got, {**policy, **change}).promote

    def naive(registry, new, old, got):
        # the candidate's own test accuracy against the one the incumbent was registered with
        return registry.get(new)["metrics"]["accuracy"] > registry.get(old)["metrics"]["accuracy"]

    return {"naive": naive, "paired": paired(penalty=1.0, switch=0.0, class_rows=NEVER),
            "penalty": paired(switch=0.0, class_rows=NEVER),
            "guard": paired(penalty=1.0, switch=0.0), "default": paired()}


def run(job):
    """one scenario through the loop, and every candidate it made judged by every gate and shadow.

    job is (kind, seed, about). returns (kind, seed, rows): one dict per
    candidate and shadow, with what each gate decided, and the truth at the step
    the gate looks: gain, the true accuracy of the candidate minus the
    incumbent's, and worst, the lowest change in recall of any class.
    """
    kind, seed, about = job
    s = Scenario.draw(seed, kind)
    registry = Registry(Store(":memory:"))
    policy = load()
    loop = Loop(registry, s, about, policy)
    replay(loop.service, s, LABELING, range(WINDOW, STEPS), [loop.update])
    out = []
    for r in loop.reviews:
        new, old = r["candidate"], r["incumbent"]
        for shadow in SHADOWS:
            look = registry.get(new)["step"] + shadow
            _, got = compare(registry, new, old, look)
            a, b = (judge(s, from_bytes(registry.store.get(m)).predict, look) for m in (new, old))
            out.append({"shadow": shadow, "gain": a["accuracy"] - b["accuracy"],
                        "worst": float(np.nanmin(a["recall"] - b["recall"])),
                        **{name: bool(gate(registry, new, old, got))
                           for name, gate in gates(policy).items()}})
    return kind, seed, out


def table(runs, tolerance):
    """per kind, gate and shadow, how its calls compare with the truth.

    promoted is the share of candidates promoted. false is the share of the
    truly worse ones promoted, missed the share of the truly better ones not
    promoted, gave the share of those giving up a class, recall falling by
    more than tolerance, that were promoted. left is the mean accuracy the
    gate leaves against one that knows the truth, max(gain, 0) - gain if
    promoted else max(gain, 0). nan where nothing fits.
    """
    names = list(gates(load()))
    out = {}
    for kind in KINDS:
        rows = [r for k, _, rs in runs if k == kind for r in rs]
        for name in names:
            for shadow in SHADOWS:
                mine = [r for r in rows if r["shadow"] == shadow]
                gain = np.array([r["gain"] for r in mine])
                went = np.array([r[name] for r in mine], dtype=bool)
                lost = np.array([r["worst"] < -tolerance for r in mine], dtype=bool)
                out[kind, name, shadow] = {
                    "candidates": len(mine), "promoted": _mean(went),
                    "false": _mean(went[gain < 0]), "missed": _mean(~went[gain > 0]),
                    "gave": _mean(went[lost]),
                    "left": _mean(np.maximum(gain, 0) - np.where(went, gain, 0)),
                }
    return out


def _mean(a):
    return float(a.mean()) if a.size else np.nan


def write(out, path):
    columns = ["candidates", "promoted", "false", "missed", "gave", "left"]
    lines = ["kind,gate,shadow," + ",".join(columns)]
    for (kind, name, shadow), row in out.items():
        cells = [str(row[c]) if isinstance(row[c], int) else f"{row[c]:.4f}" for c in columns]
        lines.append(f"{kind},{name},{shadow}," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def plot(out, path):
    kinds = [k for k in KINDS if k != "none"] + ["none"]
    names = list(gates(load()))
    fig, axes = panels([1] * len(kinds), height=10)
    x = np.arange(len(names))
    for ax, kind in zip(axes, kinds):
        for i, (shadow, color) in enumerate(zip(SHADOWS, SERIES)):
            left = [1000 * out[kind, name, shadow]["left"] for name in names]
            ax.bar(x + (i - 1.5) * 0.2, left, width=0.2, color=color, label=f"shadow {shadow}")
        style(ax, kind)
    legend(axes[0], len(SHADOWS))
    axes[-1].set_xticks(x, names)
    axes[-1].set_xlabel("gate", color=MUTED)
    save(fig, "accuracy left against a gate that knows the truth, in thousandths, "
              "mean over the candidates of each kind", path)


def report(out):
    print(f"\nat shadow {load()['shadow']}: share promoted, of the truly worse promoted, of the truly "
          f"better missed, of those giving up a class promoted, accuracy left x1000")
    print(f"{'kind':10s} {'gate':8s} {'n':>4s} {'promoted':>9s} {'false':>6s} {'missed':>7s} "
          f"{'gave':>6s} {'left':>6s}")
    for (kind, name, shadow), r in out.items():
        if shadow != load()["shadow"]:
            continue
        cells = [f"{r[c]:.0%}" if not np.isnan(r[c]) else "-" for c in ("promoted", "false",
                                                                        "missed", "gave")]
        print(f"{kind:10s} {name:8s} {r['candidates']:4d} {cells[0]:>9s} {cells[1]:>6s} "
              f"{cells[2]:>7s} {cells[3]:>6s} {1000 * r['left']:6.1f}")
    print("\naccuracy left x1000 by shadow, default gate")
    print(f"{'kind':10s}" + "".join(f"{f'shadow {w}':>12s}" for w in SHADOWS))
    for kind in KINDS:
        cells = "".join(f"{1000 * out[kind, 'default', w]['left']:12.1f}" for w in SHADOWS)
        print(f"{kind:10s}" + cells)


def main():
    parser = argparse.ArgumentParser(description="measure the gate against the truth")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/gate"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    jobs = [(kind, FIRST + i, about) for kind in KINDS for i in range(args.seeds)]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = pool.map(run, jobs, chunksize=4)
    out = table(runs, load()["tolerance"])

    print(f"{args.seeds} seeds per kind from {FIRST}, "
          f"{sum(len(rs) for _, _, rs in runs) // len(SHADOWS)} candidates from the loop")
    report(out)
    args.out.mkdir(parents=True, exist_ok=True)
    write(out, args.out / "gates.csv")
    plot(out, args.out / "gates.png")
    stamp(args.out, args, about)
    print(f"wrote gates.csv, gates.png and about.json to {args.out}")


if __name__ == "__main__":
    main()
