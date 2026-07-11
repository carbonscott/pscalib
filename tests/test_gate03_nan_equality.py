#!/usr/bin/env python3
"""GATE-03 regression: the cross-check's CALIB equality must be NaN-aware.

Background
----------
``stress/stress_xcheck.py`` cross-checks psdata+pscalib calibrated output against
psana. jungfrau (and friends) mask bad/undefined pixels to ``NaN`` in the
calibrated frame. The original gate compared calib arrays with a *bare*
``np.array_equal(mcal, pcal)``, which returns ``False`` whenever ANY ``NaN`` is
present (because ``NaN != NaN``). So a byte-for-byte identical calib array
(matching NaN positions included) would spuriously FAIL that predicate -- yet the
campaign recorded PASS, and ``bench_index.py``'s ``eq_calib`` gate (which IS
NaN-aware) reported PASS on the SAME detector/run. Two gates, contradictory
predicates, both "PASS": one record was wrong. GATE-03.

The fix makes the CALIB predicate NaN-aware (``np.array_equal(a, b,
equal_nan=True)``, matching ``bench_index``) while keeping the RAW predicate bare
byte-exact (raw is integer data with no NaN -- a stray NaN there is a real
divergence we must still catch).

This test is self-contained: stdlib + numpy only, no psana, no psdata, no SLAC
data. It imports ``stress_xcheck.py`` directly (its ``main()`` is guarded by
``__name__ == '__main__'``, so importing it never touches psana/psdata) and both
(a) exercises the calib/raw equality helpers at runtime and (b) inspects the
source via AST.

Discriminator
-------------
On the PARENT commit (``origin/main``) ``stress/stress_xcheck.py`` is UNTRACKED,
so a clean worktree of the parent does not contain it -> the existence assertion
below FAILS. On the fix commit the script is tracked AND NaN-aware -> PASS. A
committed-but-still-plain-``array_equal`` version would also fail, via the
runtime NaN-aware assertion.
"""

import ast
import importlib.util
import os

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
# cwd-robust: locate the artifact relative to THIS file, not the process cwd.
SCRIPT = os.path.normpath(
    os.path.join(_HERE, os.pardir, "stress", "stress_xcheck.py")
)


def _load_script_module():
    """Import ``stress_xcheck.py`` as a module (asserts it exists first).

    Importing runs only the top-level code (imports + ``def``s); ``main()`` is
    guarded by ``if __name__ == '__main__'`` and the module name here is not
    ``__main__``, so no psana/psdata import is triggered.
    """
    assert os.path.isfile(SCRIPT), (
        "stress/stress_xcheck.py is missing at %s -- on the PARENT commit it is "
        "UNTRACKED, so this gate is expected to FAIL there; the fix commits it."
        % SCRIPT
    )
    spec = importlib.util.spec_from_file_location("stress_xcheck_gate03", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_source_tree():
    """Return (source_text, parsed_ast) for the artifact (asserts existence)."""
    assert os.path.isfile(SCRIPT), (
        "stress/stress_xcheck.py is missing at %s -- expected on the PARENT "
        "(untracked); the fix commits it." % SCRIPT
    )
    with open(SCRIPT, "r") as fh:
        src = fh.read()
    return src, ast.parse(src)


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _func_defs(tree):
    return {
        n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    }


def _array_equal_calls(node):
    """All ``[np.]array_equal(...)`` Call nodes under ``node``."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else None
            )
            if name == "array_equal":
                out.append(n)
    return out


def _has_equal_nan_true(call):
    for kw in call.keywords:
        if kw.arg == "equal_nan":
            v = kw.value
            return isinstance(v, ast.Constant) and v.value is True
    return False


def _references_isnan(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "isnan":
            return True
        if isinstance(n, ast.Name) and n.id == "isnan":
            return True
    return False


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_script_exists_and_parses():
    """The artifact must exist (cwd-robust) and be valid Python.

    This is the discriminator: absent on the untracked parent -> FAIL there.
    """
    src, tree = _read_source_tree()
    assert isinstance(tree, ast.Module)
    assert "eq_calib" in src, "expected an eq_calib helper in the artifact"
    print("[gate-03] stress_xcheck.py present at %s and parses" % SCRIPT)


def test_calib_helper_is_nan_aware():
    """The CALIB equality helper treats matching NaN positions as EQUAL.

    A plain ``np.array_equal`` would return False on identical-with-NaN input --
    that was the bug. It must still DISCRIMINATE a real (non-NaN) difference.
    """
    mod = _load_script_module()
    eq_calib = getattr(mod, "eq_calib", None)
    assert callable(eq_calib), "stress_xcheck must expose an eq_calib(a, b) helper"

    a = np.array([[1.0, np.nan], [np.nan, 2.0]])
    b = a.copy()  # identical INCLUDING matching NaN positions
    assert eq_calib(a, b), (
        "eq_calib must be NaN-aware: identical arrays with matching NaN "
        "positions must compare EQUAL (a bare np.array_equal returns False here "
        "-- that is the GATE-03 bug)"
    )

    # still discriminating: a real difference in a non-NaN element -> UNEQUAL.
    c = a.copy()
    c[0, 0] = 9.0
    assert not eq_calib(a, c), (
        "eq_calib must still catch a genuine (non-NaN) divergence"
    )
    print("[gate-03] eq_calib is NaN-aware and still discriminating")


def test_raw_helper_stays_bare_byte_exact():
    """The RAW equality helper must stay bare byte-exact (NOT NaN-aware).

    Raw is integer data with no NaN; a stray NaN there is a real divergence.
    Bare np.array_equal reports identical-with-NaN as UNEQUAL, which is exactly
    the strict behavior we must preserve for raw.
    """
    mod = _load_script_module()
    eq_raw = getattr(mod, "eq_raw", None)
    assert callable(eq_raw), "stress_xcheck must expose an eq_raw(a, b) helper"

    a = np.array([[1.0, np.nan], [np.nan, 2.0]])
    b = a.copy()
    assert not eq_raw(a, b), (
        "eq_raw must stay bare byte-exact (no equal_nan): identical-with-NaN "
        "must compare UNEQUAL, so a NaN sneaking into a raw frame is caught. "
        "If this fails, raw was wrongly made NaN-aware."
    )
    # sanity: bare equality on plain (NaN-free) data still works.
    assert eq_raw(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert not eq_raw(np.array([1, 2, 3]), np.array([1, 2, 4]))
    print("[gate-03] eq_raw stays bare byte-exact (strict, no equal_nan)")


def test_ast_calib_nan_aware_raw_bare():
    """Source-level proof: CALIB predicate uses equal_nan=True; RAW stays bare."""
    _src, tree = _read_source_tree()
    funcs = _func_defs(tree)
    assert "eq_calib" in funcs, "eq_calib function definition not found"
    assert "eq_raw" in funcs, "eq_raw function definition not found"

    calib_calls = _array_equal_calls(funcs["eq_calib"])
    assert calib_calls, "eq_calib must call np.array_equal"
    calib_nan_aware = (
        any(_has_equal_nan_true(c) for c in calib_calls)
        or _references_isnan(funcs["eq_calib"])
    )
    assert calib_nan_aware, (
        "eq_calib's comparison must be NaN-aware: np.array_equal(a, b, "
        "equal_nan=True) (or an isnan-aware form) -- matching bench_index"
    )

    raw_calls = _array_equal_calls(funcs["eq_raw"])
    assert raw_calls, "eq_raw must call np.array_equal"
    raw_bare = (
        not any(_has_equal_nan_true(c) for c in raw_calls)
        and not any(
            any(kw.arg == "equal_nan" for kw in c.keywords) for c in raw_calls
        )
        and not _references_isnan(funcs["eq_raw"])
    )
    assert raw_bare, (
        "eq_raw's comparison must stay BARE byte-exact (no equal_nan / no isnan)"
    )
    print("[gate-03] AST confirms: CALIB is NaN-aware, RAW stays bare")


def main():
    print("=" * 72)
    print("GATE-03: stress_xcheck CALIB equality is NaN-aware; RAW stays bare")
    print("=" * 72)
    test_script_exists_and_parses()
    test_calib_helper_is_nan_aware()
    test_raw_helper_stays_bare_byte_exact()
    test_ast_calib_nan_aware_raw_bare()
    print("\nALL GATE-03 CHECKS PASSED")


if __name__ == "__main__":
    main()
