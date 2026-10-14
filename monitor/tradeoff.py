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
from spine.provenance import code, versions
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS

SIGNALS = {"inputs": InputDrift, "predictions": PredictionDrift, "errors": Performance,
           "labels": LabelShares}
CHANGES = ("covariate", "concept", "noise", "new")
THRESHOLDS = np.arange(2.0, 40.01, 0.5)
HORIZON = 200
FIRST = 10_000


def paths(job):
    """one scenario served from WINDOW to STEPS - 1 with every monitor watching.

    job is (kind, seed, about). returns (kind, seed, at, {signal: path}).
    """
    kind, seed, about = job
    s = Scenario.draw(seed, kind)
    registry = Registry(Store(":memory:"))
    model = train(registry, s, WINDOW, None, about, LABELING)
    monitors = {name: watch(registry, model) for name, watch in SIGNALS.items()}
    replay(Service(registry, model), s, LABELING, range(WINDOW, STEPS),
           [m.update for m in monitors.values()])
    return kind, seed, s.at, {name: np.array(m.path) for name, m in monitors.items()}


def alarms(path, thresholds):
    """index of the first value of path over each threshold, len(path) when there's none."""
    return np.searchsorted(np.maximum.accumulate(path), thresholds, side="right")


def tradeoff(runs, thresholds):
    """per signal and threshold, how often it fires for nothing and how well it catches changes.

    false is the share of none runs with any alarm. for each change, detected
    is the share of runs whose first alarm comes at the change or within
    HORIZON steps after it (one before it is a false alarm, not a detection),
    and delay the median steps from change to alarm among those, nan if none.
    """
    out = {}
    for name in SIGNALS:
        row = {}
        first = [(kind, at, alarms(p[name], thresholds), len(p[name])) for kind, _, at, p in runs]
        row["false"] = np.mean([a < n for kind, _, a, n in first if kind == "none"], axis=0)
        for change in CHANGES:
            hit = [(a - (at - WINDOW)) for kind, at, a, _ in first if kind == change]
            if not hit:
                row[f"{change} detected"] = row[f"{change} delay"] = np.full(len(thresholds), np.nan)
                continue
            delays = np.array(hit, dtype=float)
            caught = (delays >= 0) & (delays < HORIZON)
            row[f"{change} detected"] = caught.mean(axis=0)
            row[f"{change} delay"] = np.array([np.median(d[c]) if c.any() else np.nan
                                               for d, c in zip(delays.T, caught.T)])
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
    columns = ["false"] + [f"{c} {what}" for c in CHANGES for what in ("detected", "delay")]
    lines = ["signal,threshold," + ",".join(c.replace(" ", "_") for c in columns)]
    for name, row in table.items():
        for i, h in enumerate(thresholds):
            lines.append(f"{name},{h:g}," + ",".join(f"{row[c][i]:.4f}" for c in columns))
    path.write_text("\n".join(lines) + "\n")


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
    pick = operating(table, args.budget)

    print(f"{args.seeds} seeds per kind from {FIRST}, each served for {STEPS - WINDOW} steps; "
          f"thresholds keeping false alarms in at most {args.budget:.0%} of unchanged runs")
    print(f"{'signal':12s} {'threshold':>9s} {'false':>6s}"
          + "".join(f"{c:>16s}" for c in CHANGES))
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
    args.out.mkdir(parents=True, exist_ok=True)
    write(table, THRESHOLDS, args.out / "tradeoff.csv")
    plot(table, THRESHOLDS, args.budget, args.out / "tradeoff.png")
    print(f"wrote {args.out / 'tradeoff.csv'} and {args.out / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
