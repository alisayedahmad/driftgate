import argparse
import json
from pathlib import Path

import numpy as np

from charts import MUTED, SERIES, panels, save, style
from eval.harness import GENUINE, HURT, POLICIES
from spine.provenance import code, stamp, versions
from world.scenarios import KINDS

METRICS = ("false promotion", "missed promotion", "detection delay", "false alarm",
           "recovery", "wasted retrains", "regret")


def _rate(runs, part, whole):
    total = sum(r[whole] for r in runs)
    return sum(r[part] for r in runs) / total if total else np.nan


def _median(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.median(values)) if values else np.nan


def metrics(runs):
    """the seven numbers for a set of runs.

    false promotion is the share of promotions of a model truly worse than the
    one it replaced, missed promotion the share of candidates kept back that
    were truly better, both by more than GENUINE. detection delay is the median
    steps from the change to the trigger's first decision after it, false alarm
    the share of decisions with the model in service not HURT worse than when
    it started. recovery is the median steps from the change to the accuracy
    coming back near where it was. wasted retrains is the share of candidates
    not truly better by GENUINE. regret is the mean accuracy times steps below
    the best model trained by then. undetected and unrecovered are the shares
    of runs with a change that got no decision after it, or never came back.
    """
    changed = [r for r in runs if r["at"] is not None]
    return {
        "runs": len(runs),
        "false promotion": _rate(runs, "worse promoted", "promotions"),
        "missed promotion": _rate(runs, "better kept", "kept"),
        "detection delay": _median([r["delay"] for r in changed]),
        "undetected": float(np.mean([np.isnan(r["delay"]) for r in changed])) if changed else np.nan,
        "false alarm": _rate(runs, "false alarms", "decisions"),
        "recovery": _median([r["recovery"] for r in changed]),
        "unrecovered": (float(np.mean([np.isnan(r["recovery"]) for r in changed])) if changed
                        else np.nan),
        "wasted retrains": _rate(runs, "wasted", "candidates"),
        "regret": float(np.mean([r["regret"] for r in runs])),
    }


def table(runs):
    """metrics() per policy over every kind, then per policy and kind, where there are runs."""
    out = {(name, "all"): metrics([r for r in runs if r["policy"] == name]) for name in POLICIES
           if any(r["policy"] == name for r in runs)}
    for kind in KINDS:
        for name in POLICIES:
            mine = [r for r in runs if r["policy"] == name and r["kind"] == kind]
            if mine:
                out[name, kind] = metrics(mine)
    return out


def write(out, path):
    columns = ["runs", *METRICS, "undetected", "unrecovered"]
    lines = ["policy,kind," + ",".join(c.replace(" ", "_") for c in columns)]
    for (name, kind), row in out.items():
        cells = [str(row[c]) if isinstance(row[c], int) else f"{row[c]:.4f}" for c in columns]
        lines.append(f"{name},{kind}," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def plot(out, path):
    kinds = [k for k in KINDS if k != "none"] + ["none"]
    fig, axes = panels([1] * len(kinds), height=10)
    x = np.arange(len(POLICIES))
    top = max(out[name, kind]["regret"] for name in POLICIES for kind in kinds) * 1.1
    for ax, kind in zip(axes, kinds):
        ax.bar(x, [out[name, kind]["regret"] for name in POLICIES], width=0.6,
               color=[SERIES[1] if name == "default" else SERIES[0] for name in POLICIES])
        ax.set_ylim(0, top)
        style(ax, kind)
    axes[-1].set_xticks(x, list(POLICIES))
    axes[-1].set_xlabel("policy", color=MUTED)
    save(fig, "regret: accuracy times steps below the best model any policy had trained by then, "
              "mean per run", path)


def report(out):
    print(f"{'policy':14s} {'false prom':>10s} {'missed':>7s} {'delay':>6s} {'false al':>9s} "
          f"{'recovery':>9s} {'wasted':>7s} {'regret':>7s}")
    for name in POLICIES:
        r = out[name, "all"]
        print(f"{name:14s} {r['false promotion']:10.0%} {r['missed promotion']:7.0%} "
              f"{r['detection delay']:6.0f} {r['false alarm']:9.0%} {r['recovery']:9.0f} "
              f"{r['wasted retrains']:7.0%} {r['regret']:7.2f}")
    print("\nregret per kind (accuracy x steps per run)")
    print(f"{'policy':14s}" + "".join(f"{k:>11s}" for k in KINDS))
    for name in POLICIES:
        print(f"{name:14s}" + "".join(f"{out[name, k]['regret']:11.2f}" for k in KINDS))


def main():
    parser = argparse.ArgumentParser(description="the seven metrics per policy, from make sweep")
    parser.add_argument("--runs", type=Path, default=Path("results/runs/runs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/eval"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    runs = [json.loads(line) for line in args.runs.read_text().splitlines()]
    out = table(runs)
    print(f"{len(runs)} runs from {args.runs}; genuinely better or worse means by more than "
          f"{GENUINE:g}, an alarm has something behind it past a {HURT:g} drop")
    report(out)
    args.out.mkdir(parents=True, exist_ok=True)
    write(out, args.out / "policies.csv")
    plot(out, args.out / "regret.png")
    stamp(args.out, args, about)
    print(f"wrote policies.csv, regret.png and about.json to {args.out}")


if __name__ == "__main__":
    main()
