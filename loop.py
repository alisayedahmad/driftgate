import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from charts import MUTED, SERIES, mark, panel, panels, save
from gate.compare import compare
from gate.policy import decide, load, review
from monitor.tradeoff import SIGNALS
from retrain.train import TEST, WINDOW, check, train
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
    step the window plans, and the gate looks at it policy["shadow"] steps
    later. while nothing is pending, a challenger is also trained every
    policy["challenge"] steps on the WINDOW steps before, and judged the same
    way. a promotion serves the candidate from the next step on, with fresh
    monitors and trigger, and policy["rollback"] steps later the model it
    replaced goes before the gate against it, on the rows it served since: if
    the old one would be promoted, it serves again and is watched at once.
    after a hold, the trigger's or the gate's, the same model is watched again
    policy["retry"] steps later, twice as late each time after that. a model's
    candidates are cut from the earliest start any of its alarms gave, or the
    alarms of a model rolled back from it; a challenger kept back changes
    nothing. gate judges a candidate and is called like gate.policy.review,
    the default; the rollback check always compares rows. events lists (step, what
    happened), and reviews every candidate the gate looked at, as dicts of
    step, candidate, incumbent, challenger and promote.
    """

    def __init__(self, registry, s, about, policy, window=Window(), trigger="informed",
                 gate=review):
        self.registry, self.s, self.about = registry, s, about
        self.policy, self.window, self.kind, self.gate = policy, window, trigger, gate
        self.service = Service(registry, train(registry, s, WINDOW, None, about, LABELING))
        self.events, self.reviews = [], []
        self.promoted(WINDOW)

    def promoted(self, step, replaced=None):
        self.fallback = None if replaced is None else {"model": replaced, "at": step,
                                                       "since": self.since}
        self.since, self.retry = None, self.policy["retry"]
        self.watch(step)

    def watch(self, step):
        self.monitors = {name: m(self.registry, self.service.model) for name, m in SIGNALS.items()}
        self.trigger = Trigger(self.monitors, self.kind)
        self.plan = self.candidate = None
        self.challenger, self.challenge = False, step + self.policy["challenge"]

    def rest(self, step):
        # re-armed at once, the monitors find the same change again within steps
        self.trigger, self.resume = None, step + self.retry
        self.retry *= 2

    def update(self, step, log):
        back = self.fallback
        if back is not None and step == back["at"] + self.policy["rollback"] and self.undone(step):
            return
        if self.trigger is None:
            if step < self.resume:
                return
            self.watch(step)
        for m in self.monitors.values():
            m.update(step, log)
        self.trigger.update(step, log)
        if self.plan is None and self.trigger.decision is not None:
            self.decided(step, self.trigger.decision)
        if self.plan is None and step >= self.challenge:
            # the default window, the last WINDOW steps
            self.plan, self.challenger = (step, None), True
        if self.plan is not None and step == self.plan[0]:
            self.candidate = train(self.registry, self.s, step, self.service.model, self.about,
                                   LABELING, self.plan[1])
            first, stop = self.plan[1]["fit"] if self.plan[1] else (step - WINDOW, step - TEST)
            who = "challenger" if self.challenger else "candidate"
            self.events.append((step, f"trained {who} {self.candidate[:12]} on steps "
                                      f"{first}-{stop - 1}"))
        if self.candidate is not None and step == self.plan[0] + self.policy["shadow"]:
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
        _, _, got, verdict = self.gate(self.registry, self.candidate, self.service.model,
                                       self.policy)
        self.events.append((step, f"{'promote' if verdict.promote else 'keep'}: gain "
                                  f"{got['gain']:+.3f} ± {got['se']:.3f} on {got['rows']} rows, "
                                  f"{verdict.reason}"))
        self.reviews.append({"step": step, "candidate": self.candidate,
                             "incumbent": self.service.model, "challenger": self.challenger,
                             "promote": verdict.promote})
        if verdict.promote:
            replaced = self.service.model
            self.service.switch(self.registry, self.candidate)
            return self.promoted(step, replaced)
        if self.challenger:
            self.plan = self.candidate = None
            self.challenger, self.challenge = False, step + self.policy["challenge"]
            return
        self.rest(step)

    def undone(self, step):
        # roles swapped: the model that was replaced is the candidate now
        back, self.fallback = self.fallback, None
        _, got = compare(self.registry, back["model"], self.service.model, step, back["at"] + 1)
        verdict = decide(got, self.policy)
        since = f"{got['gain']:+.3f} ± {got['se']:.3f} on {got['rows']} rows since"
        if not verdict.promote:
            self.events.append((step, f"stay: the model it replaced does {since}"))
            return False
        self.events.append((step, f"roll back to {back['model'][:12]}: it does {since}, "
                                  f"{verdict.reason}"))
        self.service.switch(self.registry, back["model"])
        # not a hold: whatever the new model's alarms found, the old one is in it too
        found = [t for t in (back["since"], self.since) if t is not None]
        self.since = min(found) if found else None
        self.watch(step)
        return True


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
