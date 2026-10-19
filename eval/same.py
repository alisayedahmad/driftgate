import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

from threadpoolctl import threadpool_limits

from eval import decisions
from eval.harness import FIRST, run
from retrain.test_train import ABOUT, rows
from retrain.train import CONFIG, fit, to_bytes
from world import fingerprint
from world.scenarios import KINDS


def main():
    parser = argparse.ArgumentParser(
        description="write what this environment computes, to compare it byte for byte with another")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "world.txt").write_text(fingerprint.table())
    (args.out / "decisions.txt").write_text(decisions.table())
    fitted = hashlib.sha256(to_bytes(fit(*rows(3), CONFIG))).hexdigest()
    (args.out / "fit.txt").write_text(fitted + "\n")
    # one seed per kind under every policy, a small make sweep
    jobs = [(kind, FIRST, ABOUT) for kind in KINDS]
    with mp.get_context("fork").Pool(4, initializer=threadpool_limits, initargs=(1,)) as pool:
        runs = [r for rs in pool.map(run, jobs, chunksize=1) for r in rs]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in runs))


if __name__ == "__main__":
    main()
