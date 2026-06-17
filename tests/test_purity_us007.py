#!/usr/bin/env python3
"""US-007 acceptance test: the import-purity guard prefix-matches ``psana.*``.

CONTEXT (2026-06-16 dependency audit).  ``assert_no_framework_imports()``
underwrites the whole project's "no framework" guarantee, but it used to match
forbidden modules by EXACT name (``m in sys.modules``).  The real compiled xtc2
reader is the SUBMODULE ``psana.dgram`` -- it lands in ``sys.modules`` under the
dotted key ``"psana.dgram"``, never as a bare leaf ``"dgram"``.  So the
``"dgram"`` token in ``FORBIDDEN_MODULES`` could NEVER fire, and the guard had
teeth only via the ``"psana"`` token.  US-007 makes the guard prefix-match:
a module leaks if its name EQUALS a forbidden root OR ``startswith(root + ".")``.

Acceptance criteria covered:

  (1) PUBLIC TUPLE STABLE.  ``pscalib.FORBIDDEN_MODULES`` is unchanged in value
      and shape -- the asserts that pin it elsewhere still hold.

  (2) NEW TEETH (deterministic, no psana).  Using a forbidden root whose parent
      package is ABSENT from ``sys.modules`` (so the bare-name token can't fire),
      the OLD exact-name logic PASSES (misses the submodule) while the NEW
      prefix logic RAISES.  This isolates exactly the gap US-007 closes.

  (3) POSITIVE CONTROL (prod psana env).  In a fresh interpreter that sources
      psconda.sh and does ``import psana.dgram``, ``assert_no_framework_imports``
      RAISES, and the error names ``psana.dgram`` specifically -- proving the
      submodule key (not just the bare ``psana`` token) is detected.

  (4) OFFLINE PATH UNCHANGED.  Import + reload + apply stays
      ('psana','mpi4py','h5py','dgram','pymongo')-clean (in-proc and in a fresh
      interpreter).

The positive control (3) needs the PRODUCTION psana env (psconda.sh) and is
skipped if psana is not importable.  Everything else runs without psana.
"""

import os
import subprocess
import sys

import numpy as np

# --- locate the pscalib package (parent of this tests dir) ------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# The whole-import forbidden set (shared with pscalib._purity).  Pinned here so
# this test fails loudly if the public tuple ever silently changes shape.
FORBIDDEN = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# (1) the public tuple is stable -- value AND shape
# --------------------------------------------------------------------------
def test_forbidden_tuple_stable():
    import pscalib
    assert pscalib.FORBIDDEN_MODULES == FORBIDDEN, pscalib.FORBIDDEN_MODULES
    # shape: a 5-tuple of plain strings (the asserts elsewhere pin this exact
    # representation; US-007 changes only the MATCHING logic, never the tuple).
    assert isinstance(pscalib.FORBIDDEN_MODULES, tuple)
    assert all(isinstance(x, str) for x in pscalib.FORBIDDEN_MODULES)
    assert len(pscalib.FORBIDDEN_MODULES) == 5
    print("[ok] (1) FORBIDDEN_MODULES tuple value+shape unchanged")


# --------------------------------------------------------------------------
# (2) the new teeth, demonstrated deterministically WITHOUT psana
# --------------------------------------------------------------------------
def test_prefix_match_catches_submodule_old_missed():
    """Inject a fake forbidden submodule whose PARENT is absent, so only a
    prefix match can see it.  The OLD exact-name rule passes (the gap); the NEW
    rule raises (the teeth)."""
    import types
    import pscalib._purity as purity

    # A root that is NOT a real importable package -> its bare name will never
    # be in sys.modules on its own, isolating the submodule-only case.
    root = "pscalib_us007_fakeframework"
    submod = root + ".reader"            # mimics 'psana.dgram'
    assert root not in sys.modules and submod not in sys.modules

    # Simulate the leak: only the SUBMODULE key is present (as Python would do
    # if a C-extension submodule were imported but we only inspected leaf names).
    sys.modules[submod] = types.ModuleType(submod)
    try:
        forbidden = (root,)

        # OLD logic (exact name match) -- what the guard used to do.  It does
        # NOT see the submodule: this is precisely the bug US-007 fixes.
        old_leaked = [m for m in forbidden if m in sys.modules]
        assert old_leaked == [], (
            "old exact-name logic should MISS the submodule (that was the gap)")

        # NEW logic via the real helper -- it DOES see the submodule.
        new_leaked = purity._leaked_modules(forbidden)
        assert submod in new_leaked, new_leaked

        # ...and the public assert RAISES on it.
        raised = False
        try:
            purity.assert_no_framework_imports(forbidden=forbidden)
        except AssertionError as e:
            raised = True
            assert submod in str(e), str(e)
        assert raised, "assert_no_framework_imports must raise on the submodule leak"
    finally:
        sys.modules.pop(submod, None)

    # The bare root itself must still be caught (equality arm of the rule).
    sys.modules[root] = types.ModuleType(root)
    try:
        try:
            purity.assert_no_framework_imports(forbidden=(root,))
            raise AssertionError("must raise on the bare-root leak too")
        except AssertionError as e:
            assert root in str(e), str(e)
    finally:
        sys.modules.pop(root, None)

    print("[ok] (2) prefix match catches a submodule the old exact check missed")


def test_prefix_match_no_false_positive():
    """A module that merely SHARES a forbidden prefix without the dot boundary
    (e.g. 'psanafoo') must NOT be flagged -- the rule is 'root' or 'root.*',
    never a bare ``startswith(root)``."""
    import types
    import pscalib._purity as purity

    decoy = "psanafoo"                   # NOT 'psana' nor 'psana.*'
    assert decoy not in sys.modules
    sys.modules[decoy] = types.ModuleType(decoy)
    try:
        purity.assert_no_framework_imports()   # default FORBIDDEN incl. 'psana'
    finally:
        sys.modules.pop(decoy, None)
    print("[ok] (2b) 'psanafoo' is not a false positive (dot-boundary respected)")


# --------------------------------------------------------------------------
# (4) offline path stays clean -- in-proc and fresh interpreter
# --------------------------------------------------------------------------
def test_offline_import_purity_in_proc():
    import pscalib
    pscalib.assert_no_framework_imports()
    for m in FORBIDDEN:
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"
    # no psana.* / mpi4py.* submodule keys either
    bad = [k for k in sys.modules
           if any(k == r or k.startswith(r + ".") for r in FORBIDDEN)]
    assert not bad, bad
    print("[ok] (4) offline import purity (in-proc), prefix-clean")


def test_offline_apply_purity_subprocess():
    """Fresh interpreter: import pscalib + run a jungfrau apply on synthetic
    arrays.  The prefix-aware guard must stay clean; numpy must be present."""
    script = (
        "import sys\n"
        "import numpy as np\n"
        "import pscalib\n"
        "ped = np.zeros((3, 32, 512, 1024), np.float32)\n"
        "gain = np.ones((3, 32, 512, 1024), np.float32)\n"
        "raw = np.zeros((32, 512, 1024), np.uint16)\n"
        "out = pscalib.calib('jungfrau_raw_0_1_0', raw,\n"
        "                    {'pedestals': ped, 'pixel_gain': gain})\n"
        "assert out.shape == (32, 512, 1024) and out.dtype == np.float32, "
        "(out.shape, out.dtype)\n"
        "pscalib.assert_no_framework_imports()\n"
        "bad = [k for k in sys.modules\n"
        "       if any(k == r or k.startswith(r + '.') for r in %r)]\n"
        "assert not bad, bad\n"
        "assert 'numpy' in sys.modules\n"
        "print('CLEAN')\n"
        % (FORBIDDEN,)
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout
    print("[ok] (4) offline apply import purity (subprocess), prefix-clean")


# --------------------------------------------------------------------------
# (3) POSITIVE CONTROL -- needs the production psana env (psconda.sh)
# --------------------------------------------------------------------------
def test_positive_control_psana_dgram():
    """In a FRESH interpreter, deliberately ``import psana.dgram`` and assert the
    guard now RAISES and names the SUBMODULE ``psana.dgram`` -- the teeth the old
    exact-name check lacked for that key.  Also confirm, in the same process,
    that the OLD exact-name logic would NOT have flagged ``psana.dgram``."""
    script = (
        "import sys\n"
        "import pscalib\n"
        "import psana.dgram\n"           # the real compiled xtc2 reader submodule
        "assert 'psana.dgram' in sys.modules\n"
        # OLD exact-name logic against the FULL forbidden tuple: it never had a
        # 'psana.dgram' token, so it could not flag that specific key.  (It would
        # catch the bare 'psana' the parent import also pulls -- that is NOT the
        # gap; the gap is the dead 'dgram' leaf token.)
        "old_dgram_hit = 'psana.dgram' in set(pscalib.FORBIDDEN_MODULES) and "
        "'psana.dgram' in sys.modules\n"
        "assert not old_dgram_hit, 'sanity: no literal psana.dgram token exists'\n"
        # NEW guard: must RAISE, and the message must name the submodule key.
        "raised = False\n"
        "try:\n"
        "    pscalib.assert_no_framework_imports()\n"
        "except AssertionError as e:\n"
        "    raised = True\n"
        "    msg = str(e)\n"
        "    assert 'psana.dgram' in msg, msg\n"
        "assert raised, 'guard must RAISE after import psana.dgram'\n"
        # And the helper reports the submodule key explicitly.
        "leaked = pscalib._purity._leaked_modules(pscalib.FORBIDDEN_MODULES)\n"
        "assert 'psana.dgram' in leaked, leaked\n"
        "print('RAISED')\n"
    )
    # Unlike the offline subprocesses, this one MUST keep psana importable, so
    # PREPEND _PKG_PARENT to the inherited PYTHONPATH (which carries the prod
    # psana env from psconda.sh) rather than replacing it.
    inherited = os.environ.get("PYTHONPATH", "")
    pp = _PKG_PARENT + (os.pathsep + inherited if inherited else "")
    env = dict(os.environ, PYTHONPATH=pp)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, (out.stdout + "\n" + out.stderr)
    assert "RAISED" in out.stdout, out.stdout
    print("[ok] (3) positive control: import psana.dgram -> guard RAISES on "
          "'psana.dgram'")


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-007 acceptance: import-purity guard prefix-matches psana.*")
    print("=" * 72)

    # offline / deterministic checks always run (no psana needed)
    test_forbidden_tuple_stable()
    test_prefix_match_catches_submodule_old_missed()
    test_prefix_match_no_false_positive()
    test_offline_import_purity_in_proc()
    test_offline_apply_purity_subprocess()

    if not _have_psana():
        print("\n[skip] psana not importable -- positive control (import "
              "psana.dgram) skipped.  Source psconda.sh on sdfiana025.")
        print("\nUS-007 offline checks PASSED (psana-dependent gate skipped)")
        return

    test_positive_control_psana_dgram()

    print("\nALL US-007 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
