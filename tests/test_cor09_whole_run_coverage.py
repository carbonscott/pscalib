"""COR-09 regression: the calib/image cross-check must cover the WHOLE run.

``stress/stress_xcheck.py`` cross-checks psdata+pscalib against psana on a
compute node (needs psana + SLAC data), so this meta-test does NOT run that leg.
It is self-contained (stdlib + numpy, no psana, no SLAC data).

The COR-09 defect: the calib byte-exactness gate sampled only a HEAD PREFIX --
events ``0 .. N-1`` from the FRONT (the ``--stride`` flag was never once passed,
so the loop decimated nothing and stopped after the first ~100 events).  That is
structurally blind to every known live divergence in this project, because they
are ALL positional and LATE: the chunk roll at k=37,120 (STR-01), the step-2
config override (CAL-02), the ragged tail at k=17,872 (FAIL-01).

Unlike GATE-01/03, ``stress_xcheck.py`` is ALREADY TRACKED on the parent
(``origin/main`` = f679ecb2e73184bf85fd2c7d33e66ed3ab201caa) WITH the
head-prefix bug, so the discriminator here is BEHAVIORAL, not existence-based:

  * We obtain the set of positions the gate would cross-check.  On the FIX that
    is the importable helper ``gen_positions`` (whole-run + hazard sample).  On
    the PARENT there is NO whole-run helper -- its positions are the inline head
    prefix ``0 .. min(nevents, N)-1``, which we model faithfully.
  * We then assert whole-run coverage on those positions: they must REACH THE
    LAST EVENT (``max == n_events-1``), SPAN THE WHOLE RUN (into the final
    quarter), and INCLUDE the LATE HAZARDS (+/-1 around a given BeginStep and a
    given chunk roll).

  PARENT  -> positions are the head prefix (max ~ 99 of 17,872)   -> FAIL.
  FIX     -> positions span the whole run and hit the hazards     -> PASS.

Also asserts (AST/grep) that the ``--positions`` flag / hazard-list mechanism
exists in the source (absent on the parent, present on the fix).
"""
import ast
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.normpath(
    os.path.join(HERE, os.pardir, "stress", "stress_xcheck.py"))

# jungfrau mfx100848724/r51 canonical L1 count (FAIL-01 ragged-tail run).
N_EVENTS = 17872
# The calib campaign sampled 100 events from the FRONT -> head prefix max ~ 99.
N_SAMPLE = 100
# A LATE BeginStep boundary (CAL-02 step-2 override bites here) and a LATE
# chunk-roll boundary (STR-01) -- both far past the head-prefix cap.
STEP_BOUNDARY = 8000
CHUNK_BOUNDARY = 12000


def _read_source():
    with open(SCRIPT, "r") as fh:
        return fh.read()


def _load_module():
    """Import stress_xcheck.py as a module (numpy-only; its psdata/psana imports
    are lazy inside ``main`` and the ``__main__`` block does not execute)."""
    assert os.path.isfile(SCRIPT), (
        f"stress_xcheck.py not found at {SCRIPT} -- it is TRACKED on the parent, "
        f"so this path must always resolve.")
    spec = importlib.util.spec_from_file_location("stress_xcheck_cor09", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _calib_positions(mod, n_events=N_EVENTS, n=N_SAMPLE,
                     step_boundaries=None, chunk_boundaries=None):
    """The set of positions the CALIB cross-check would visit.

    FIX: the importable whole-run helper ``gen_positions`` (preferred) or
    ``resolve_positions``.  PARENT: no such helper exists -- the gate's inline
    loop checks the first ``min(n, n_events)`` present events (stride defaults to
    1, never passed), i.e. a HEAD PREFIX ``0 .. min(n, n_events)-1``.  Modelling
    that prefix is what makes this a BEHAVIORAL parent-vs-fix discriminator:
    the whole-run assertions below then fail on the parent's real behavior.
    """
    if hasattr(mod, "gen_positions"):
        return list(mod.gen_positions(
            n_events, n=n,
            step_boundaries=step_boundaries,
            chunk_boundaries=chunk_boundaries))
    if hasattr(mod, "resolve_positions"):
        return list(mod.resolve_positions(
            n_events, nevents=n, stride=1, positions=None,
            step_boundaries=step_boundaries,
            chunk_boundaries=chunk_boundaries))
    # PARENT head-prefix behavior: events 0 .. min(n, n_events)-1 from the FRONT.
    return list(range(min(int(n), int(n_events))))


def test_script_exists_and_is_valid_python():
    """stress_xcheck.py is tracked on the parent, so it must always be present
    and syntactically valid (guards the import path for the behavioral tests)."""
    assert os.path.isfile(SCRIPT), f"stress_xcheck.py not found at {SCRIPT}"
    ast.parse(_read_source())  # raises SyntaxError if invalid


def test_calib_positions_reach_last_event_and_span_whole_run():
    """The calib cross-check positions must reach the LAST event and span the
    whole run -- NOT the front-100 head prefix (COR-09).

    PARENT: modeled head prefix -> max ~ 99 -> FAILS.
    FIX: gen_positions -> max == n_events-1 -> PASSES.
    """
    mod = _load_module()
    pos = _calib_positions(mod)
    assert pos, "no cross-check positions generated"

    # REACHES THE LAST EVENT -- the head-prefix bug capped this at ~99.
    assert max(pos) == N_EVENTS - 1, (
        f"calib positions must reach the last event {N_EVENTS - 1}, got "
        f"max={max(pos)} -- head-prefix regression (front ~{N_SAMPLE} of "
        f"{N_EVENTS}).")
    assert min(pos) == 0, "calib positions must include the first event (k=0)"
    # SPANS THE WHOLE RUN -- not confined to a head prefix.
    assert any(p > N_EVENTS // 2 for p in pos), (
        "calib positions confined to a head prefix -- none past the midpoint.")
    assert max(pos) > 3 * N_EVENTS // 4, (
        "no calib positions in the FINAL QUARTER -- coverage is a head prefix.")


def test_calib_positions_include_late_hazards():
    """The calib cross-check must probe +/-1 around a LATE BeginStep boundary
    (CAL-02 step override) and a LATE chunk-roll boundary (STR-01) -- both far
    past the head prefix.

    PARENT: head prefix ignores hazards -> FAILS.
    FIX: gen_positions probes +/-1 around each -> PASSES.
    """
    mod = _load_module()
    pos = set(_calib_positions(
        mod, step_boundaries=[STEP_BOUNDARY], chunk_boundaries=[CHUNK_BOUNDARY]))

    for label, b in (("BeginStep", STEP_BOUNDARY), ("chunk-roll", CHUNK_BOUNDARY)):
        assert {b - 1, b, b + 1}.issubset(pos), (
            f"calib positions do not probe +/-1 around the LATE {label} hazard "
            f"at k={b} (missing {sorted({b - 1, b, b + 1} - pos)}) -- the "
            f"head-prefix gate never reaches it.")


def test_positions_reach_last_across_run_sizes():
    """The whole-run helper reaches the last event for a range of run sizes."""
    mod = _load_module()
    if not hasattr(mod, "gen_positions"):
        # PARENT: no whole-run helper -> the property cannot hold (head prefix).
        raise AssertionError(
            "no gen_positions helper -- positions are the inline head prefix, "
            "which cannot reach the last event for large runs (COR-09).")
    for n_events, n in [(17872, 100), (73800, 200), (1000, 64), (5, 8), (1, 4)]:
        pos = mod.gen_positions(n_events, n=n)
        assert pos, f"no positions for n_events={n_events}"
        assert max(pos) == n_events - 1, (
            f"n_events={n_events}: max={max(pos)} != last {n_events - 1}")
        assert min(pos) == 0, f"n_events={n_events}: first event k=0 missing"


def test_stride_reaches_tail_not_just_a_prefix():
    """The corrected ``--stride`` must reach the tail: a raw stride decimates a
    PREFIX (coverage 0,S,..,(nevents-1)*S), so the tail is reached only if
    ``nevents*stride >= n_events`` (GATE-04).  ``stride_positions`` bumps the
    effective stride accordingly and always includes the last event.

    PARENT: no ``stride_positions`` helper -> FAILS.
    FIX: the helper reaches the tail -> PASSES.
    """
    mod = _load_module()
    assert hasattr(mod, "stride_positions"), (
        "no stride_positions helper -- the --stride decimation still stops at a "
        "head prefix (nevents*stride < n_events) instead of reaching the tail.")
    # Even a tiny nevents with a small stride must still reach the last event.
    pos = mod.stride_positions(N_EVENTS, nevents=10, stride=1)
    assert max(pos) == N_EVENTS - 1, (
        f"stride decimation did not reach the tail: max={max(pos)} != "
        f"{N_EVENTS - 1}")
    # nevents * effective-stride must span the whole run.
    assert any(p > 3 * N_EVENTS // 4 for p in pos), \
        "strided positions confined to a head prefix"


def test_positions_flag_and_hazard_mechanism_exist():
    """The source must expose an explicit ``--positions`` flag AND a hazard-list
    mechanism (BeginStep + chunk-roll boundary discovery).

    PARENT: neither exists -> FAILS.  FIX: both exist -> PASSES.
    """
    src = _read_source()

    assert "--positions" in src, (
        "no explicit --positions flag -- a caller cannot request the "
        "hazard-targeted list / an arbitrary position set (COR-09).")

    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "gen_positions" in funcs, "no whole-run gen_positions helper"
    assert "step_boundary_positions" in funcs, (
        "no step_boundary_positions helper -- BeginStep (CAL-02) hazards are "
        "not targeted.")
    assert "chunk_boundary_positions" in funcs, (
        "no chunk_boundary_positions helper -- chunk-roll (STR-01/FAIL-01) "
        "hazards are not targeted.")

    # the head-prefix decimation `if i % args.stride: continue` must be gone.
    assert "i % args.stride" not in src, (
        "the inline `if i % args.stride: continue` head-prefix decimation is "
        "still present -- COR-09 not fixed.")


def test_nan_aware_eq_calib_and_psana_oracle_preserved():
    """GATE-03's NaN-aware eq_calib and the psana-oracle structure must be
    preserved by the COR-09 fix (do NOT revert them)."""
    mod = _load_module()
    # GATE-03: NaN-aware calib equality (matching NaN positions are equal).
    assert hasattr(mod, "eq_calib"), "eq_calib predicate removed"
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert mod.eq_calib(a, b), "eq_calib is no longer NaN-aware (GATE-03 revert)"
    assert not mod.eq_calib(a, np.array([1.0, np.nan, 4.0]))
    # RAW equality must stay STRICT (no equal_nan).
    assert hasattr(mod, "eq_raw")
    assert not mod.eq_raw(np.array([1.0, np.nan]), np.array([1.0, np.nan]))

    # psana-oracle structure preserved: the source still drives psana events and
    # feeds psana's own _calibconst into pscalib.calib (BYO path).
    src = _read_source()
    for token in ("psrun.events()", "det.raw.raw(evt)", "det.raw.calib(evt)",
                  "_calibconst", "pscalib.calib"):
        assert token in src, f"psana-oracle structure lost: '{token}' missing"


def main():
    test_script_exists_and_is_valid_python()
    test_calib_positions_reach_last_event_and_span_whole_run()
    test_calib_positions_include_late_hazards()
    test_positions_reach_last_across_run_sizes()
    test_stride_reaches_tail_not_just_a_prefix()
    test_positions_flag_and_hazard_mechanism_exist()
    test_nan_aware_eq_calib_and_psana_oracle_preserved()
    print("COR-09 whole-run coverage regression: all checks PASS")


if __name__ == "__main__":
    main()
