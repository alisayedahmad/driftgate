from eval.decisions import FILE, table


def test_the_loop_decides_what_it_decided_before():
    expected = FILE.read_text().splitlines()
    got = table().splitlines()
    moved = [f"{old} -> {new.split()[-1]}" for old, new in zip(expected, got) if old != new]
    assert len(got) == len(expected)
    assert not moved, ("the loop decides differently, run make decisions if that was meant:\n"
                       + "\n".join(moved))
