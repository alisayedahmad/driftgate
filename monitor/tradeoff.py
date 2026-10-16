import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from charts import MUTED, SERIES, legend, mark, panels, save, style
from monitor.input_drift import InputDrift
from monitor.performance import LabelShares, Performance
from monitor.prediction_drift import PredictionDrift
from retrain.train import WINDOW, train
from serving.app import Service, replay
from spine.provenance import code, stamp, versions
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS

SIGNALS = {"inputs": InputDrift, "predictions": PredictionDrift, "errors": Performance,
           "labels": LabelShares}
CHANGES = ("covariate", "concept", "noise", "new")
STATS = ("detected", "delay", "start", "start q10", "start q90")
THRESHOLDS = np.arange(2.0, 40.01, 0.5)
HORIZON = 200
FIRST = 10_000


def paths(job):
    """one scenario served from WINDOW to STEPS - 1 with every monitor watching.

    job is (kind, seed, about). returns (kind, seed, at, {signal: path},
    {signal: starts}), starts as floats with nan where a monitor had none.
    """
    kind, seed, about = job
    s = Scenario.draw(seed, kind)
    registry = Registry(Store(":memory:"))
    model = train(registry, s, WINDOW, None, about, LABELING)
    monitors = {name: watch(registry, model) for name, watch in SIGNALS.items()}
    replay(Service(registry, model), s, LABELING, range(WINDOW, STEPS),
           [m.update for m in monitors.values()])
    return (kind, seed, s.at, {name: np.array(m.path) for name, m in monitors.items()},
            {name: np.array(m.starts, dtype=float) for name, m in monitors.items()})


def alarms(path, thresholds):
    """index of the first value of path over each threshold, len(path) when there's none."""
    return np.searchsorted(np.maximum.accumulate(path), thresholds, side="right")


def among(values, caught, q):
    # per threshold, quantile q of values over the runs caught at that threshold
    return np.array([np.quantile(v[c], q) if c.any() else np.nan for v, c in zip(values.T, caught.T)])


def tradeoff(runs, thresholds):
    """per signal and threshold, how often it fires for nothing and how well it catches changes.

    false is the share of none runs with any alarm. for each change, detected
    is the share of runs whose first alarm comes at the change or within
    HORIZON steps after it (one before it is a false alarm, not a detection),
    and delay the median steps from change to alarm among those. start is the
    median steps from the change to where the monitor put its start at the
    alarm, negative when it put it before, and start q10 and start q90 the
    10th and 90th percentiles, among the same runs. nan when there are none.
    """
    out = {}
    for name in SIGNALS:
        row = {}
        first = [(kind, at, alarms(p[name], thresholds), b[name]) for kind, _, at, p, b in runs]
        row["false"] = np.mean([a < len(b) for kind, _, a, b in first if kind == "none"], axis=0)
        for change in CHANGES:
            hit = [(at, a, b) for kind, at, a, b in first if kind == change]
            if not hit:
                for what in STATS:
                    row[f"{change} {what}"] = np.full(len(thresholds), np.nan)
                continue
            delays = np.array([a - (at - WINDOW) for at, a, _ in hit], dtype=float)
            starts = np.array([np.append(b, np.nan)[a] - at for at, a, b in hit])
            caught = (delays >= 0) & (delays < HORIZON)
            row[f"{change} detected"] = caught.mean(axis=0)
            row[f"{change} delay"] = among(delays, caught, 0.5)
            row[f"{change} start"] = among(starts, caught, 0.5)
            row[f"{change} start q10"] = among(starts, caught, 0.1)
            row[f"{change} start q90"] = among(starts, caught, 0.9)
        out[name] = row
    return out


def plot(table, thresholds, budget, path):
    fig, axes = panels([1] * len(CHANGES), height=9)
    for ax, change in zip(axes, CHANGES):
        for color, (name, row) in zip(SERIES, table.items()):
            ax.plot(row["false"], row[f"{change} detected"], color=color, linewidth=1.6, label=name)
        style(ax, change)
        ax.set_ylim(-0.03, 1.03)
    legend(axes[0], len(table))
    axes[-1].set_xlim(0, 0.5)
    axes[-1].set_xlabel("share of runs without a change that raise a false alarm", color=MUTED)
    # labelled on the concept panel, the only one with nothing near its top at the budget
    mark([axes[1], axes[0], *axes[2:]], budget, f"{budget:.0%}")
    save(fig, f"share of changes caught within {HORIZON} steps, against false alarms, "
              f"over thresholds {thresholds[0]:g} to {thresholds[-1]:g}", path)


def operating(table, budget):
    """per signal, the index of the lowest threshold keeping false alarms at most budget, or None."""
    return {name: int(np.argmax(row["false"] <= budget)) if (row["false"] <= budget).any() else None
            for name, row in table.items()}


def write(table, thresholds, path):
    columns = ["false"] + [f"{c} {what}" for c in CHANGES for what in STATS]
    lines = ["signal,threshold," + ",".join(c.replace(" ", "_") for c in columns)]
    for name, row in table.items():
        for i, h in enumerate(thresholds):
            lines.append(f"{name},{h:g}," + ",".join(f"{row[c][i]:.4f}" for c in columns))
    path.write_text("\n".join(lines) + "\n")


def report(table, pick):
    print(f"{'signal':12s} {'threshold':>9s} {'false':>6s}" + "".join(f"{c:>16s}" for c in CHANGES))
    for name, row in table.items():
        i = pick[name]
        if i is None:
            print(f"{name:12s} none up to {THRESHOLDS[-1]:g}")
            continue
        cells = []
        for c in CHANGES:
            delay = row[f"{c} delay"][i]
            cells.append(f"{row[f'{c} detected'][i]:6.0%} in {'-' if np.isnan(delay) else f'{delay:.0f}':>4s}")
        print(f"{name:12s} {THRESHOLDS[i]:9g} {row['false'][i]:6.0%}" + "".join(f"{c:>16s}" for c in cells))

    print("\nwhere each caught change was put to start at those thresholds, in steps from "
          "the change: median (10th to 90th percentile)")
    print(f"{'signal':12s}" + "".join(f"{c:>21s}" for c in CHANGES))
    for name, row in table.items():
        i = pick[name]
        if i is None:
            continue
        cells = []
        for c in CHANGES:
            mid, low, high = (row[f"{c} {what}"][i] for what in ("start", "start q10", "start q90"))
            cells.append("-" if np.isnan(mid) else
                         f"{round(mid):+d} ({round(low):+d} to {round(high):+d})")
        print(f"{name:12s}" + "".join(f"{c:>21s}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="trade each monitor's false alarms against its delay")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/monitor"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    jobs = [(kind, FIRST + i, about) for kind in KINDS for i in range(args.seeds)]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = pool.map(paths, jobs, chunksize=4)
    table = tradeoff(runs, THRESHOLDS)

    print(f"{args.seeds} seeds per kind from {FIRST}, each served for {STEPS - WINDOW} steps; "
          f"thresholds keeping false alarms in at most {args.budget:.0%} of unchanged runs")
    report(table, operating(table, args.budget))
    args.out.mkdir(parents=True, exist_ok=True)
    write(table, THRESHOLDS, args.out / "tradeoff.csv")
    plot(table, THRESHOLDS, args.budget, args.out / "tradeoff.png")
    stamp(args.out, args, about)
    print(f"wrote tradeoff.csv, tradeoff.png and about.json to {args.out}")


if __name__ == "__main__":
    main()
