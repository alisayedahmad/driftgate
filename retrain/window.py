from dataclasses import dataclass

from retrain.train import WINDOW


@dataclass(frozen=True)
class Window:
    """how a candidate's rows are chosen once the trigger decides to retrain.

    the candidate is trained patience steps after the decision, on the WINDOW
    steps before that. the last gate of them are held out, to test it on and
    later to judge it against the serving model. the rest is cut margin steps
    before the estimated start of the change: rows after the cut weigh 1, and
    of the steps before it the old_share nearest the cut are kept, their rows
    weighing old_weight. old_share 0 drops the old regime, 1 keeps all of it.
    """

    margin: int = 10
    old_weight: float = 1.0
    old_share: float = 0.0
    patience: int = 20
    gate: int = 20

    def plan(self, decision):
        """(step, plan) for retrain.train: when to train, and on which steps.

        ValueError when the cut leaves no step to fit on before the held-out ones.
        """
        step = decision.step + self.patience
        first, stop = step - WINDOW, step - self.gate
        cut = max(decision.start - self.margin, first)
        if cut >= stop:
            raise ValueError(f"cut at {cut}, nothing left to fit on before the held-out steps "
                             f"from {stop}")
        first = cut - round(self.old_share * (cut - first))
        weights = [[first, cut, self.old_weight]] if first < cut and self.old_weight != 1 else []
        return step, {"fit": [first, stop], "weights": weights, "test": [stop, step]}
