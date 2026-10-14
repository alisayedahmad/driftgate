from dataclasses import dataclass

# the lowest keeping false alarms in at most 5% of unchanged runs, from make monitor
THRESHOLDS = {"inputs": 12.5, "predictions": 11.0, "errors": 12.0, "labels": 11.0}
NEW_LABELS = 30
POLICIES = ("informed", "naive")


@dataclass(frozen=True)
class Decision:
    """what to do about the serving model, decided at `step`.

    action is "retrain" or "hold", a hold meaning the labels look wrong rather
    than the model. start is the step the change is estimated to have begun
    at: the earliest start among the alarms the decision rests on.
    """

    step: int
    action: str
    reason: str
    start: int


class Trigger:
    """turns the monitors' alarms into one decision.

    monitors maps the names in THRESHOLDS to the four monitors of the serving
    model. update is a replay watcher and has to come after theirs. naive
    retrains as soon as the errors alarm. informed reads the other alarms at
    that point: retrain if the inputs or the predictions moved too, hold if
    only the label shares did, which is label noise, retrain if nothing else
    moved, which is a concept break. inputs moving on their own call for
    nothing. a label of a class the model never saw overrides all of it:
    retrain once NEW_LABELS of them are back. decision is None until there is
    one and doesn't change after.
    """

    def __init__(self, monitors, policy="informed"):
        if policy not in POLICIES:
            raise ValueError(f"policy is one of {POLICIES}, not {policy!r}")
        self.monitors = monitors
        self.policy = policy
        self.alarms = {}
        self.unseen = 0
        self.decision = None

    def update(self, step, log):
        if self.decision is not None:
            return
        for name, m in self.monitors.items():
            if name not in self.alarms and m.path[-1] > THRESHOLDS[name]:
                self.alarms[name] = m.starts[-1]
        known = self.monitors["labels"].share > 0
        self.unseen += int((~known[log.landed(step)["label"]]).sum())
        verdict = self.naive() if self.policy == "naive" else self.informed()
        if verdict is not None:
            action, reason, rests = verdict
            start = min(self.alarms[name] for name in rests if name in self.alarms)
            self.decision = Decision(step, action, reason, start)

    def naive(self):
        if "errors" in self.alarms:
            return "retrain", "the errors went up", ["errors"]
        return None

    def informed(self):
        moved = ["inputs", "predictions"]
        if self.unseen:
            if self.unseen < NEW_LABELS:
                return None
            return "retrain", f"{self.unseen} labels of a class never seen", ["labels", *moved]
        if "errors" not in self.alarms:
            return None
        if any(name in self.alarms for name in moved):
            return "retrain", "the inputs moved and the errors went up", ["errors", *moved]
        if "labels" in self.alarms:
            return "hold", "the label shares moved but the inputs didn't, label noise", ["errors", "labels"]
        return "retrain", "only the errors went up, the concept changed", ["errors"]
