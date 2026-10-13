import argparse
from pathlib import Path

import numpy as np

from world.scenarios import KINDS, Scenario
from world.stream import CLASSES, INFORMATIVE

STEPS = 1000
WINDOW = 25

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def trace(s, steps=STEPS):
    """per-step numbers for scenario s, over steps 0 to steps - 1.

    best is the mean of max P(y|x) over the step's rows, the accuracy of the
    best possible classifier. true and measured are for the old classifier,
    the argmax of the world before any change: its expected accuracy, and its
    hit rate on the observed labels of every row, as if all came back at once.
    freq is (steps, CLASSES + 1), the share of each label in the step.
    """
    best, true, measured = np.empty(steps), np.empty(steps), np.empty(steps)
    freq = np.empty((steps, CLASSES + 1))
    for t in range(steps):
        x, y = s.batch(t)
        p = s.proba(x, t)
        old = s.world.proba(x).argmax(axis=1)
        best[t] = p.max(axis=1).mean()
        true[t] = p[np.arange(len(x)), old].mean()
        measured[t] = (old == y).mean()
        freq[t] = np.bincount(y, minlength=CLASSES + 1) / len(y)
    return {"best": best, "true": true, "measured": measured, "freq": freq}


def describe(s):
    if s.kind == "none":
        return f"seed {s.seed}, no change"
    if s.kind == "covariate":
        what = f"column {np.flatnonzero(s.shift)[0]} moved by {s.shift.sum():+.2f}"
    elif s.kind == "concept":
        cos = (np.trace(s.rotation) - (INFORMATIVE - 2)) / 2
        what = f"rotated by {np.degrees(np.arccos(np.clip(cos, -1, 1))):.0f} degrees"
    elif s.kind == "noise":
        source, target = s.flip
        what = f"{s.size:.0%} of class {source} labeled {target}"
    else:
        what = f"class {CLASSES} on {s.size:.1%} of rows"
    return f"seed {s.seed}, {s.kind} at step {s.at}, size {s.size:.3f}, {what}"


def table(s, r):
    if s.at is None:
        parts = [("all", slice(None))]
    else:
        parts = [("before", slice(0, s.at)), ("after", slice(s.at, None))]
    lines = ["".ljust(22) + "".join(name.rjust(9) for name, _ in parts)]
    for key, name in [("best", "best possible"), ("true", "old model, true"),
                      ("measured", "old model, measured")]:
        lines.append(name.ljust(22) + "".join(f"{r[key][part].mean():9.3f}" for _, part in parts))
    for c in range(CLASSES + 1):
        shares = [r["freq"][part, c].mean() for _, part in parts]
        if any(shares):
            lines.append(f"class {c} share".ljust(22) + "".join(f"{v:9.3f}" for v in shares))
    return "\n".join(lines)


def rolling(a):
    # trailing mean over full windows, value i ends at step i + WINDOW - 1
    c = np.concatenate([[0.0], np.cumsum(a)])
    return (c[WINDOW:] - c[:-WINDOW]) / WINDOW


def plot(s, r, path):
    # imported here, the tests only need trace and don't pay for matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 2]})
    steps = np.arange(WINDOW - 1, len(r["best"]))
    # best goes last and thinner, it often sits exactly on true
    lines = [("old model, true", r["true"], SERIES[0], 1.6),
             ("old model, measured", r["measured"], SERIES[1], 1.6),
             ("best possible", r["best"], MUTED, 1.0)]
    _panel(top, steps, lines, "accuracy")
    shown = [c for c in range(CLASSES + 1) if r["freq"][:, c].any()]
    _panel(bottom, steps, [(f"class {c}", r["freq"][:, c], SERIES[c], 1.6) for c in shown],
           "label share")
    bottom.set_xlabel("step", color=MUTED)
    if s.at is not None:
        for ax in (top, bottom):
            ax.axvline(s.at, color=AXIS, linewidth=1, zorder=1)
        top.annotate(f"change at {s.at}", (s.at, 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
    fig.suptitle(describe(s) + f"\nmeans over the last {WINDOW} steps", color=INK, fontsize=10,
                 x=0.06, ha="left")
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def _panel(ax, steps, lines, name):
    ax.set_facecolor(SURFACE)
    for label, values, color, width in lines:
        ax.plot(steps, rolling(values), color=color, linewidth=width, label=label)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel(name, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), frameon=False, fontsize=9,
              labelcolor=INK, ncol=len(lines))
    if len(lines) <= 4:
        ends = [(rolling(values)[-1], label) for label, values, _, _ in lines]
        _label_ends(ax, steps[-1], ends)


def _label_ends(ax, x, ends):
    # push labels apart so they don't overlap, at least 6% of the axis height
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    placed = []
    for y, label in sorted(ends):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=8, annotation_clip=False)


def main():
    parser = argparse.ArgumentParser(description="draw a scenario and look at it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default="concept")
    parser.add_argument("--out", type=Path, default=Path("results/world"))
    args = parser.parse_args()

    s = Scenario.draw(args.seed, args.kind)
    r = trace(s)
    print(describe(s))
    print(table(s, r))
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.out / f"{s.kind}-{s.seed}"
    columns = np.column_stack([np.arange(STEPS), r["best"], r["true"], r["measured"], r["freq"]])
    header = "step,best,true,measured," + ",".join(f"class{c}" for c in range(CLASSES + 1))
    np.savetxt(name.with_suffix(".csv"), columns, delimiter=",", header=header, comments="",
               fmt=["%d"] + ["%.6f"] * (columns.shape[1] - 1))
    plot(s, r, name.with_suffix(".png"))
    print(f"wrote {name}.csv and {name}.png")


if __name__ == "__main__":
    main()
