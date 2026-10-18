import hashlib
from pathlib import Path

from gate.policy import load
from loop import Loop
from retrain.test_train import ABOUT
from retrain.train import WINDOW
from serving.app import replay
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import KINDS, Scenario
from world.summary import STEPS

SEED = 0
FILE = Path(__file__).with_name("decisions.txt")


def fingerprint(s):
    """first 16 hex digits of a sha256 over what the loop did on s with the default policy.

    that is every event, step and text, and the lineage of the model serving
    at the end.
    """
    loop = Loop(Registry(Store(":memory:")), s, ABOUT, load())
    replay(loop.service, s, LABELING, range(WINDOW, STEPS), [loop.update])
    h = hashlib.sha256()
    for step, what in loop.events:
        h.update(f"{step} {what}\n".encode())
    for record in loop.registry.lineage(loop.service.model):
        h.update(record["model"].encode())
    return h.hexdigest()[:16]


def table():
    return "".join(f"{kind} {SEED} {fingerprint(Scenario.draw(SEED, kind))}\n" for kind in KINDS)


if __name__ == "__main__":
    FILE.write_text(table())
    print(f"wrote {FILE}")
