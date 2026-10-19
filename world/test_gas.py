import numpy as np
import pytest

from world.gas import CACHE, FEATURES, KNOWN, Gas, fetch, ordered, parse

HAVE = (CACHE / "gas.zip").exists()


def test_a_batch_file_parses_into_rows_and_gases_from_zero():
    x, y = parse("1;10.000000 1:2.5 128:-1.0\n6 3:4.0\n\n")
    assert y.tolist() == [0, 5]
    assert x.shape == (2, FEATURES)
    assert x[0, 0] == 2.5 and x[0, 127] == -1.0 and x[1, 2] == 4.0
    assert np.count_nonzero(x) == 3


def test_an_archive_that_isnt_the_one_expected_is_refused(tmp_path):
    (tmp_path / "gas.zip").write_bytes(b"not the archive")
    with pytest.raises(ValueError):
        fetch(tmp_path)


def test_batches_keep_their_order_and_are_shuffled_inside():
    # each row carries its batch and its place in it
    parts = [(np.column_stack([np.full(n, b), np.arange(n)]).astype(float), np.arange(n) % 2)
             for b, n in enumerate([40, 30, 20, 10, 50])]
    x, y, starts = ordered(parts)
    raw = np.concatenate([p[0] for p in parts])
    known = raw[:sum(len(p[1]) for p in parts[:KNOWN])]
    back = x * known.std(axis=0) + known.mean(axis=0)
    assert starts.tolist() == [0, 40, 70, 90, 100]
    assert (np.round(back[:, 0]) == np.repeat(range(5), [40, 30, 20, 10, 50])).all()
    assert not (np.round(back[:40, 1]) == np.arange(40)).all()
    assert np.array_equal(x, ordered(parts)[0])
    assert np.allclose(x[:90].mean(axis=0), 0) and np.allclose(x[:90].std(axis=0), 1)


@pytest.mark.skipif(not HAVE, reason="the gas data isn't downloaded, make data")
def test_the_gas_stream_goes_through_the_whole_pipeline():
    from gate.policy import load
    from loop import Loop
    from retrain.test_train import ABOUT
    from retrain.train import WINDOW, check
    from serving.app import replay
    from spine.registry import Registry
    from spine.store import Store
    from world.labels import LABELING

    s = Gas()
    loop = Loop(Registry(Store(":memory:")), s, ABOUT, load())
    replay(loop.service, s, LABELING, range(WINDOW, WINDOW + 60), [loop.update])
    first = loop.service.log.models[0]
    assert check(loop.registry, first) == []
    assert loop.registry.stream(first) == s
    assert loop.service.log.batch(WINDOW)["proba"].shape == (s.rows, s.labels)
    assert len(loop.monitors["inputs"].path) > 0
