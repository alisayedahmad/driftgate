from world.fingerprint import FILE, table


def test_the_world_draws_what_it_drew_before():
    expected = FILE.read_text().splitlines()
    got = table().splitlines()
    moved = [f"{old} -> {new.split()[-1]}" for old, new in zip(expected, got) if old != new]
    assert len(got) == len(expected)
    assert not moved, "data changed, run make fingerprints if that was meant:\n" + "\n".join(moved)
