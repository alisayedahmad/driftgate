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
    """first 16 hex digits of a sha256 over what the loop decided on s with the default policy.

    that is the trigger's decisions, every review and whether it promoted,
    the rollbacks, and the steps the models serving at the end descend from
    were trained at. no model bytes and no measured numbers: another CPU can
    move those in the last bit without the loop deciding anything else.
    """
    loop = Loop(Registry(Store(":memory:")), s, ABOUT, load())
    replay(loop.service, s, LABELING, range(WINDOW, STEPS), [loop.update])
    h = hashlib.sha256()
    for d in loop.decisions:
        h.update(f"decide {d['step']} {d['action']}\n".encode())
    for r in loop.reviews:
        h.update(f"review {r['step']} {r['challenger']} {r['promote']}\n".encode())
    for step, what in loop.events:
        if what.startswith("roll back"):
            h.update(f"roll back {step}\n".encode())
    for record in loop.registry.lineage(loop.service.model):
        h.update(f"lineage {record['step']}\n".encode())
    return h.hexdigest()[:16]


def table():
    return "".join(f"{kind} {SEED} {fingerprint(Scenario.draw(SEED, kind))}\n" for kind in KINDS)


if __name__ == "__main__":
    FILE.write_text(table())
    print(f"wrote {FILE}")
