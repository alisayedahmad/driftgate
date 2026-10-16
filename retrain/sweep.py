import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from charts import MUTED, SERIES, legend, panels, save, style
from monitor.tradeoff import SIGNALS
from retrain.train import WINDOW, from_bytes, train
from retrain.trigger import POLICIES, Trigger
from retrain.window import Window
from serving.app import Service, replay
from spine.provenance import code, stamp, versions
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS
from world.truth import judge

# the thresholds were tuned on seeds from 10000, these are kept apart
FIRST = 20_000
# each knob moved on its own away from the defaults
WINDOWS = {"default": Window(), "margin 0": Window(margin=0), "margin 30": Window(margin=30),
           "slice": Window(old_share=0.25), "weighted": Window(old_share=1.0, old_weight=0.2),
           "whole": Window(old_share=1.0), "patience 10": Window(patience=10),
           "patience 40": Window(patience=40), "gate 10": Window(gate=10),
           "gate 40": Window(gate=40)}
# no margin and cut where the change really started, what a perfect estimate would give
KNOWN = "true start"
NAMES = (*WINDOWS, KNOWN)


def plans(s, decision):
    """(window name, step, plan) for each window that can be planned after decision."""
    windows = [(name, w, decision) for name, w in WINDOWS.items()]
    if s.at is not None:
        windows.append((KNOWN, replace(WINDOWS["default"], margin=0), replace(decision, start=s.at)))
    for name, w, d in windows:
        try:
            yield name, *w.plan(d)
        except ValueError:
            continue


def run(job):
    """one scenario served with both triggers watching, and a candidate per window on each retrain.

    job is (kind, seed, about). returns (kind, seed, at, decisions, candidates):
    decisions maps each policy to its Decision or None, candidates is one dict
    per policy and window with the step the candidate was trained at, its rows,
    and the accuracy judged then for it, the serving model and the best possible
    one, and worst, the lowest change in recall of any class from serving to it.
    """
    kind, seed, about = job
    s = Scenario.draw(seed, kind)
    registry = Registry(Store(":memory:"))
    serving = train(registry, s, WINDOW, None, about, LABELING)
    monitors = {name: watch(registry, serving) for name, watch in SIGNALS.items()}
    triggers = {policy: Trigger(monitors, policy) for policy in POLICIES}
    replay(Service(registry, serving), s, LABELING, range(WINDOW, STEPS),
           [m.update for m in monitors.values()] + [t.update for t in triggers.values()])

    judged = {}

    def scored(model, step):
        if (model, step) not in judged:
            predict = (from_bytes(registry.store.get(model)).predict if model != "best"
                       else lambda x: s.proba(x, step).argmax(axis=1))
            judged[model, step] = judge(s, predict, step)
        return judged[model, step]

    candidates = []
    for policy, trigger in triggers.items():
        if trigger.decision is None or trigger.decision.action != "retrain":
            continue
        for name, step, plan in plans(s, trigger.decision):
            model = train(registry, s, step, serving, about, LABELING, plan)
            new, old = scored(model, step), scored(serving, step)
            candidates.append({
                "policy": policy, "window": name, "step": step,
                "rows": json.loads(registry.store.get(registry.get(model)["trained_on"]))["rows"],
                "candidate": new["accuracy"], "serving": old["accuracy"],
                "best": scored("best", step)["accuracy"],
                "worst": float(np.nanmin(new["recall"] - old["recall"])),
            })
    return kind, seed, s.at, {p: t.decision for p, t in triggers.items()}, candidates


def decisions(runs):
    """per kind and policy: how often it retrained or held, how often too early, and how late.

    early is the share of runs deciding anything before the change, delay the
    median steps from the change to the decision among the others that decided.
    """
    out = {}
    for kind in KINDS:
        mine = [(at, d) for k, _, at, d, _ in runs if k == kind]
        if not mine:
            continue
        for policy in POLICIES:
            taken = [(at, d[policy]) for at, d in mine]
            late = [d.step - at for at, d in taken if d is not None and at is not None and d.step >= at]
            out[kind, policy] = {
                "runs": len(taken),
                "retrain": np.mean([d is not None and d.action == "retrain" for _, d in taken]),
                "hold": np.mean([d is not None and d.action == "hold" for _, d in taken]),
                "early": np.mean([d is not None and (at is None or d.step < at) for at, d in taken]),
                "delay": float(np.median(late)) if late else np.nan,
            }
    return out


def windows(runs):
    """per kind, policy and window: what the candidates bought over the serving model.

    gain is candidate minus serving accuracy, both judged at the step the
    candidate was trained at: its median, 10th and 90th percentiles. better is
    the share of candidates with a positive gain, short the median of best
    minus candidate, worst the median of the candidates' worst, rows of their
    training rows. only candidates when there were none.
    """
    out = {}
    for kind in KINDS:
        for policy in POLICIES:
            for name in NAMES:
                c = [c for k, _, _, _, cs in runs if k == kind for c in cs
                     if c["policy"] == policy and c["window"] == name]
                gain = np.array([x["candidate"] - x["serving"] for x in c])
                row = {"candidates": len(c)}
                if c:
                    row |= {"gain": np.median(gain), "gain q10": np.quantile(gain, 0.1),
                            "gain q90": np.quantile(gain, 0.9), "better": np.mean(gain > 0),
                            "short": np.median([x["best"] - x["candidate"] for x in c]),
                            "worst": np.median([x["worst"] for x in c]),
                            "rows": np.median([x["rows"] for x in c])}
                out[kind, policy, name] = row
    return out


def write(table, keys, columns, path):
    lines = [",".join(keys + columns).replace(" ", "_")]
    for key, row in table.items():
        cells = [str(row[c]) if isinstance(row.get(c), int) else f"{row.get(c, np.nan):.4f}"
                 for c in columns]
        lines.append(",".join([*key, *cells]))
    path.write_text("\n".join(lines) + "\n")


def plot(table, path):
    kinds = [k for k in KINDS if k != "none"] + ["none"]
    fig, axes = panels([1] * len(kinds), height=11)
    x = np.arange(len(NAMES))
    for ax, kind in zip(axes, kinds):
        for i, (policy, color) in enumerate(zip(POLICIES, SERIES)):
            rows = [table[kind, policy, name] for name in NAMES]
            mid = np.array([r.get("gain", np.nan) for r in rows])
            low = np.array([r.get("gain q10", np.nan) for r in rows])
            high = np.array([r.get("gain q90", np.nan) for r in rows])
            at = x + (i - 0.5) * 0.25
            ax.vlines(at, low, high, color=color, linewidth=1.4)
            ax.plot(at, mid, "o", color=color, markersize=4, label=policy)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        style(ax, kind)
    legend(axes[0], len(POLICIES))
    axes[-1].set_xticks(x, NAMES, rotation=30, ha="right")
    save(fig, "accuracy a candidate gains over the serving model, judged when it's trained: "
              "median and 10th to 90th percentile", path)


def report(decided, table):
    print(f"{'kind':10s} {'policy':9s} {'retrain':>8s} {'hold':>6s} {'early':>6s} {'delay':>6s}")
    for (kind, policy), r in decided.items():
        delay = "-" if np.isnan(r["delay"]) else f"{r['delay']:.0f}"
        print(f"{kind:10s} {policy:9s} {r['retrain']:8.0%} {r['hold']:6.0%} {r['early']:6.0%} {delay:>6s}")
    for policy in POLICIES:
        print(f"\n{policy}: median accuracy gained over the serving model (share of candidates better)")
        print(f"{'window':12s}" + "".join(f"{k:>16s}" for k in KINDS))
        for name in NAMES:
            cells = []
            for kind in KINDS:
                r = table[kind, policy, name]
                cells.append("-" if not r["candidates"] else f"{r['gain']:+.3f} ({r['better']:.0%})")
            print(f"{name:12s}" + "".join(f"{c:>16s}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="measure what each retraining policy buys")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/retrain"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    jobs = [(kind, FIRST + i, about) for kind in KINDS for i in range(args.seeds)]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = pool.map(run, jobs, chunksize=4)
    decided, table = decisions(runs), windows(runs)

    print(f"{args.seeds} seeds per kind from {FIRST}, served from step {WINDOW} to {STEPS - 1}")
    report(decided, table)
    args.out.mkdir(parents=True, exist_ok=True)
    write(decided, ["kind", "policy"], ["runs", "retrain", "hold", "early", "delay"],
          args.out / "decisions.csv")
    write(table, ["kind", "policy", "window"],
          ["candidates", "gain", "gain q10", "gain q90", "better", "short", "worst", "rows"],
          args.out / "windows.csv")
    plot(table, args.out / "windows.png")
    stamp(args.out, args, about)
    print(f"wrote decisions.csv, windows.csv, windows.png and about.json to {args.out}")


if __name__ == "__main__":
    main()
