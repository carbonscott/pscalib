"""GATE-01 regression: the random-access gate must cover the WHOLE run and
fail-closed on completeness.

The gate script ``stress/stress_randaccess.py`` cross-checks psdata's
random-access index against psana on a compute node (needs psana + SLAC data),
so this meta-test does NOT run that leg.  It is self-contained (stdlib + numpy,
no psana, no SLAC data) and locks the two properties GATE-01 was missing:

  * seek positions are spread across the WHOLE run and REACH THE LAST EVENT --
    not the head prefix (k<=299 / k<=399 of 17,872 / 73,800) the buggy gate
    used; and
  * the event-COUNT + timestamp-SET completeness comparison is a hard
    ``assert`` (fail-closed), not a bare ``print``.

Discriminator: on the PARENT commit ``stress/stress_randaccess.py`` is
UNTRACKED, so a clean worktree of the parent does not contain it -> the
existence / import assertions below FAIL.  On the fix (committed + corrected)
they PASS.  The script is located by a path relative to ``__file__`` so the
test is cwd-robust.
"""
import ast
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.normpath(
    os.path.join(HERE, os.pardir, "stress", "stress_randaccess.py"))


def _read_source():
    with open(SCRIPT, "r") as fh:
        return fh.read()


def _load_module():
    """Import the gate script as a module (numpy-only; its psdata/psana imports
    are lazy inside ``run`` and its ``__main__`` block does not execute here)."""
    assert os.path.isfile(SCRIPT), (
        f"stress_randaccess.py not found at {SCRIPT} -- the GATE-01 gate script "
        f"must be committed (it is UNTRACKED on the parent).")
    spec = importlib.util.spec_from_file_location(
        "stress_randaccess_gate01", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists_and_is_valid_python():
    """The gate script must be committed to the repo and syntactically valid.

    This is the discriminator: absent (untracked) on the parent -> FAIL;
    present on the fix -> PASS."""
    assert os.path.isfile(SCRIPT), (
        f"stress_randaccess.py not found at {SCRIPT} -- must be committed "
        f"(untracked on the parent).")
    ast.parse(_read_source())  # raises SyntaxError if invalid


def test_positions_reach_last_event_and_span_whole_run():
    """gen_positions must reach the LAST event and span the whole run (not a
    head prefix), with the intended count."""
    mod = _load_module()
    assert hasattr(mod, "gen_positions"), \
        "position-generation helper gen_positions is missing"
    n_events = 17872  # jungfrau mfx100848724/r51 canonical L1 count
    N = 300
    pos = mod.gen_positions(n_events, n=N)

    assert list(pos) == sorted(pos), "positions must be sorted ascending"
    assert len(set(pos)) == len(pos), "positions must be distinct"
    # REACHES THE LAST EVENT -- the head-prefix bug capped this at k=299.
    assert max(pos) == n_events - 1, (
        f"positions must reach the last event {n_events - 1}, "
        f"got max={max(pos)} (head-prefix regression)")
    assert min(pos) == 0, "positions must include the first event (k=0)"
    # SPANS THE WHOLE RUN -- not confined to a head prefix.
    assert any(p > n_events // 2 for p in pos), (
        "positions confined to a head prefix -- none above n_events/2")
    assert max(pos) > 3 * n_events // 4, "no positions in the final quarter"
    # intended count: linspace endpoints ARE 0 and n_events-1, so exactly N.
    assert len(pos) == N, f"expected {N} positions, got {len(pos)}"


def test_positions_reach_last_across_sizes_and_probe_hazards():
    """The last event is reached for a range of run sizes / N, and supplied
    chunk-roll boundaries are probed at +/-1."""
    mod = _load_module()
    for n_events, N in [(17872, 300), (73800, 400), (1000, 64), (5, 8), (1, 4)]:
        pos = mod.gen_positions(n_events, n=N)
        assert pos, f"no positions generated for n_events={n_events}"
        assert max(pos) == n_events - 1, (
            f"n_events={n_events} N={N}: max={max(pos)} != last {n_events - 1}")
        assert min(pos) == 0, f"n_events={n_events}: first event k=0 missing"
    # boundary hazards land +/-1 around a supplied roll position
    pos = mod.gen_positions(1000, n=16, boundaries=[500])
    assert {499, 500, 501}.issubset(set(pos)), \
        "chunk-roll boundary hazards (+/-1) are not probed"


def test_completeness_comparison_is_asserted_not_printed():
    """The count + timestamp-SET completeness comparison must be a hard ASSERT
    (fail-closed), not a bare print -- the original GATE-01 defect."""
    src = _read_source()
    tree = ast.parse(src)

    assert_segments = [
        ast.get_source_segment(src, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]

    # a timestamp-SET equality assert (set(...ts...) == set(...))
    assert any(("set(" in s and "ts" in s and "==" in s)
               for s in assert_segments), (
        "no `assert set(...ts...) == set(...)` found -- the timestamp-SET "
        "equality must be ASSERTED (fail-closed), not printed.")
    # an event-COUNT equality assert
    assert any(("==" in s and ("count" in s or "n_events" in s or "nfwd" in s))
               for s in assert_segments), (
        "no event-COUNT equality assert (forward == index == psana) found.")

    # the ts-SET / count comparison must NOT be merely printed
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("print(") and "set(" in stripped and "==" in stripped:
            raise AssertionError(
                "timestamp-set comparison appears inside a print(); GATE-01 "
                "requires it to be an assert (fail-closed), not printed.")


if __name__ == "__main__":
    # dual-mode: runnable as a plain script as well as under pytest
    test_script_exists_and_is_valid_python()
    test_positions_reach_last_event_and_span_whole_run()
    test_positions_reach_last_across_sizes_and_probe_hazards()
    test_completeness_comparison_is_asserted_not_printed()
    print("GATE-01 coverage regression: all checks PASS")
