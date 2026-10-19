import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from charts import MUTED, SERIES, legend, mark, panels, save, style
from eval.harness import POLICIES
from eval.oracle import best_available
from gate.policy import load, review
from loop import Loop
from retrain.train import WINDOW, train
from retrain.window import Window
from serving.app import Service, replay
from spine.provenance import code, stamp, versions
from spine.registry import Registry
from spine.store import Store
from world.gas import Gas
from world.labels import LABELING
from world.truth import realized

EVERY = 10
# the first model served as is, the naive pipeline, the defaults, none of them tuned on this data
NAMES = ("never", "naive", "default")


def served(s, about, name):
    """one run on the gas stream: the registry, and the true accuracy of what served each step."""
    registry = Registry(Store(":memory:"))
    if name == "never":
        service, watch = Service(registry, train(registry, s, WINDOW, None, about, LABELING)), []
        loop = None
    else:
        setup = POLICIES[name]
        loop = Loop(registry, s, about, {**load(), **setup.get("policy", {})},
                    window=setup.get("window", Window()), trigger=setup.get("trigger", "informed"),
                    gate=setup.get("gate", review))
        service, watch = loop.service, [loop.update]
    accuracy = []

    def truth(step, log):
        accuracy.append(realized(s, service.estimator.predict, step, 1))

    replay(service, s, LABELING, range(WINDOW, s.steps), [truth, *watch])
    return registry, loop, np.array(accuracy)


def run(job):
    """every policy on the gas stream with label arrivals from seed. job is (seed, about).

    returns {policy: record}: the mean true accuracy served, the regret against
    the best model any of them had trained by each block of EVERY steps, the
    trigger's decisions and holds, the promotions, the rollbacks, and the
    accuracy of every block.
    """
    seed, about = job
    s = Gas(seed=seed)
    runs = {name: served(s, about, name) for name in NAMES}
    blocks = np.arange(WINDOW, s.steps, EVERY)
    best, _ = best_available([r for r, _, _ in runs.values()], blocks,
                             lambda predict, t: realized(s, predict, t, EVERY))
    out = {}
    for name, (_, loop, accuracy) in runs.items():
        sizes = np.diff(np.append(blocks, s.steps))
        per_block = np.add.reduceat(accuracy, blocks - WINDOW) / sizes
        out[name] = {
            "accuracy": float(accuracy.mean()),
            "regret": float(((best - per_block) * sizes).sum()),
            "decisions": len(loop.decisions) if loop else 0,
            "holds": sum(d["action"] == "hold" for d in loop.decisions) if loop else 0,
            "promotions": sum(r["promote"] for r in loop.reviews) if loop else 0,
            "rollbacks": sum(w.startswith("roll back") for _, w in loop.events) if loop else 0,
            "blocks": per_block,
        }
    return seed, out


def table(runs):
    """per policy, the mean of each number over the seeds."""
    keys = ("accuracy", "regret", "decisions", "holds", "promotions", "rollbacks")
    return {name: {k: float(np.mean([r[name][k] for _, r in runs])) for k in keys} for name in NAMES}


def plot(runs, path):
    s = Gas()
    blocks = np.arange(WINDOW, s.steps, EVERY)
    fig, ax = panels([1], height=4.2)
    for name, color in zip(NAMES, [MUTED, SERIES[0], SERIES[1]]):
        mean = np.mean([r[name]["blocks"] for _, r in runs], axis=0)
        ax.plot(blocks, mean, color=color, linewidth=1.6, label=name)
    style(ax, "true accuracy")
    legend(ax, len(NAMES))
    ax.set_xlabel("step", color=MUTED)
    starts = [t for t in s.starts() if t > WINDOW]
    mark([ax], starts[0], "a new batch")
    for t in starts[1:]:
        ax.axvline(t, color=MUTED, linewidth=0.6, linestyle=":")
    save(fig, f"the gas sensor stream, true accuracy on every label over blocks of {EVERY} steps, "
              f"mean over {len(runs)} seeds of label arrivals", path)


def main():
    parser = argparse.ArgumentParser(description="run the pipeline on the gas sensor data")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/real"))
    args = parser.parse_args()

    about = {"code": code(), "versions": versions()}
    jobs = [(seed, about) for seed in range(args.seeds)]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    out = table(runs)

    print(f"the gas stream from step {WINDOW} to {Gas().steps - 1}, {args.seeds} seeds of label "
          f"arrivals; regret in accuracy times steps against the best model any policy had")
    print(f"{'policy':8s} {'accuracy':>9s} {'regret':>8s} {'decisions':>10s} {'holds':>6s} "
          f"{'promoted':>9s} {'rolled':>7s}")
    for name, r in out.items():
        print(f"{name:8s} {r['accuracy']:9.3f} {r['regret']:8.2f} {r['decisions']:10.1f} "
              f"{r['holds']:6.1f} {r['promotions']:9.1f} {r['rollbacks']:7.1f}")
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["policy," + ",".join(next(iter(out.values())))]
    lines += [f"{name}," + ",".join(f"{v:.4f}" for v in r.values()) for name, r in out.items()]
    (args.out / "policies.csv").write_text("\n".join(lines) + "\n")
    plot(runs, args.out / "accuracy.png")
    stamp(args.out, args, about)
    print(f"wrote policies.csv, accuracy.png and about.json to {args.out}")


if __name__ == "__main__":
    main()
