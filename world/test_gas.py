import numpy as np
import pytest

from world.gas import FEATURES, fetch, parse


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
