from monitor.tradeoff import SIGNALS
from retrain.test_train import ABOUT
from retrain.train import WINDOW, train
from retrain.trigger import NEW_LABELS, POLICIES, Trigger
from serving.app import Service, replay
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario
from world.stream import CLASSES

AT = WINDOW + 40
STEPS = range(WINDOW, AT + 200)


def decided(s):
    registry = Registry(Store(":memory:"))
    model = train(registry, s, WINDOW, None, ABOUT, LABELING)
    monitors = {name: watch(registry, model) for name, watch in SIGNALS.items()}
    triggers = {policy: Trigger(monitors, policy) for policy in POLICIES}
    service = Service(registry, model)
    replay(service, s, LABELING, STEPS,
           [m.update for m in monitors.values()] + [t.update for t in triggers.values()])
    return service.log, triggers["informed"], triggers["naive"]


def test_nothing_is_decided_without_a_change():
    _, informed, naive = decided(Scenario(3))
    assert informed.decision is None and naive.decision is None


def test_both_retrain_on_a_concept_break():
    _, informed, naive = decided(Scenario(3, "concept", at=AT, size=0.25))
    assert informed.decision.action == naive.decision.action == "retrain"
    assert AT < informed.decision.step == naive.decision.step
    assert informed.decision.start == naive.decision.start


def test_only_the_informed_policy_holds_on_label_noise():
    # on seed 4 the label shares have moved by the time the errors alarm
    _, informed, naive = decided(Scenario(4, "noise", at=AT, size=0.4))
    assert informed.decision.action == "hold"
    assert naive.decision.action == "retrain"
    assert informed.decision.step == naive.decision.step


def test_inputs_moving_alone_call_for_nothing():
    _, informed, naive = decided(Scenario(5, "covariate", at=AT, size=0.6))
    assert "inputs" in informed.alarms and "errors" not in informed.alarms
    assert informed.decision is None and naive.decision is None


def test_a_covariate_shift_that_hurts_is_dated_from_the_inputs():
    _, informed, naive = decided(Scenario(4, "covariate", at=AT, size=1.0))
    assert informed.decision.action == "retrain"
    assert abs(informed.decision.start - AT) <= 2
    assert naive.decision.start > AT + 10


def test_a_new_class_waits_for_enough_of_its_labels():
    log, informed, _ = decided(Scenario(3, "new", at=AT, size=0.05))
    step = informed.decision.step
    back = [(log.as_of(t)["label"] == CLASSES).sum() for t in (step - 1, step)]
    assert informed.decision.action == "retrain"
    assert back[0] < NEW_LABELS <= back[1]
