import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from charts import MUTED, SERIES, mark, panel, panels, save
from gate.policy import load, review
from monitor.tradeoff import SIGNALS
from retrain.train import WINDOW, check, train
from retrain.trigger import Trigger
from retrain.window import Window
from serving.app import Service, replay
from spine.provenance import code, versions
from spine.registry import fresh
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS, describe
from world.truth import judge

EVERY = 10


class Loop:
    """the whole pipeline on one scenario: serve, watch, retrain, gate, promote.

    it serves a model trained at WINDOW, and update is a replay watcher for the
    steps after that. each step the serving model's monitors update, then its
    trigger. once the trigger says to retrain, a candidate is trained at the
    step the window plans, and the gate looks at it policy["wait"] steps later.
    a promotion serves the candidate from the next step on, with fresh monitors
    and trigger. after a hold, the trigger's or the gate's, the same model is
    watched again policy["retry"] steps later, twice as late each time after
    that, and its candidates are cut from the earliest start any of its alarms
    gave. events lists (step, what happened).
    """

    def __init__(self, registry, s, about, policy, window=Window(), trigger="informed"):
        self.registry, self.s, self.about = registry, s, about
        self.policy, self.window, self.kind = policy, window, trigger
        self.service = Service(registry, train(registry, s, WINDOW, None, about, LABELING))
        self.events = []
        self.promoted()

    def promoted(self):
        self.since, self.retry = None, self.policy["retry"]
        self.watch()

    def watch(self):
        self.monitors = {name: m(self.registry, self.service.model) for name, m in SIGNALS.items()}
        self.trigger = Trigger(self.monitors, self.kind)
        self.plan = self.candidate = None

    def rest(self, step):
        # re-armed at once, the monitors find the same change again within steps
        self.trigger, self.resume = None, step + self.retry
        self.retry *= 2

    def update(self, step, log):
        if self.trigger is None:
            if step < self.resume:
                return
            self.watch()
        for m in self.monitors.values():
            m.update(step, log)
        self.trigger.update(step, log)
        if self.plan is None and self.trigger.decision is not None:
            self.decided(step, self.trigger.decision)
        if self.plan is not None and step == self.plan[0]:
            self.candidate = train(self.registry, self.s, step, self.service.model, self.about,
                                   LABELING, self.plan[1])
            first, stop = self.plan[1]["fit"]
            self.events.append((step, f"trained {self.candidate[:12]} on steps {first}-{stop - 1}"))
        if self.candidate is not None and step == self.plan[0] + self.policy["wait"]:
            self.judged(step)

    def decided(self, step, d):
        # monitors watched again only see a change from when they restarted
        self.since = d.start if self.since is None else min(self.since, d.start)
        self.events.append((step, f"{d.action}: {d.reason}, change put at {self.since}"))
        if d.action == "hold":
            return self.rest(step)
        try:
            self.plan = self.window.plan(replace(d, start=self.since))
        except ValueError as e:
            self.events.append((step, f"no window: {e}"))
            self.rest(step)

    def judged(self, step):
        _, _, got, verdict = review(self.registry, self.candidate, self.service.model, self.policy)
        self.events.append((step, f"{'promote' if verdict.promote else 'keep'}: gain "
                                  f"{got['gain']:+.3f} ± {got['se']:.3f} on {got['rows']} rows, "
                                  f"{verdict.reason}"))
        if verdict.promote:
            self.service.switch(self.registry, self.candidate)
            return self.promoted()
        self.rest(step)


def plot(s, steps, served, best, promoted, path):
    fig, ax = panels([1], height=3.6)
    panel(ax, steps, [("serving", served, SERIES[0], 1.6), ("best possible", best, MUTED, 1.2)],
          "true accuracy")
    ax.set_xlabel("step", color=MUTED)
    if s.at is not None:
        mark([ax], s.at, f"change at {s.at}")
    for step in promoted:
        ax.axvline(step, color=SERIES[1], linewidth=1, linestyle=":")
    save(fig, describe(s) + "\njudged on fresh rows every "
              f"{EVERY} steps, dotted where a candidate was promoted", path)


def main():
    parser = argparse.ArgumentParser(description="run the whole pipeline on one scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default="concept")
    parser.add_argument("--out", type=Path, default=Path("results/loop"))
    args = parser.parse_args()

    s = Scenario.draw(args.seed, args.kind)
    registry = fresh(f"loop-{s.kind}-{s.seed}")
    loop = Loop(registry, s, {"code": code(), "versions": versions()}, load())
    steps, served, best = [], [], []

    def truth(step, log):
        if step % EVERY == 0:
            steps.append(step)
            served.append(judge(s, loop.service.estimator.predict, step)["accuracy"])
            best.append(judge(s, lambda x: s.proba(x, step).argmax(axis=1), step)["accuracy"])

    replay(loop.service, s, LABELING, range(WINDOW, STEPS), [loop.update, truth])
    loop.service.log.save(registry.store.db)

    print(describe(s))
    for step, what in loop.events:
        print(f"{step:5d}  {what}")
    lineage = registry.lineage(loop.service.model)
    print("serving " + " <- ".join(f"{r['model'][:12]} (step {r['step']})" for r in lineage))
    problems = [m["model"][:12] for m in lineage if check(registry, m["model"])]
    print("check   " + ("every model in the lineage rebuilds from its record" if not problems
                        else f"these don't rebuild: {problems}"))
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.out / f"{s.kind}-{s.seed}"
    promoted = [step for step, what in loop.events if what.startswith("promote")]
    plot(s, np.array(steps), served, best, promoted, name.with_suffix(".png"))
    print(f"wrote {name}.png, the registry and the log are in {registry.store.path}")


if __name__ == "__main__":
    main()
