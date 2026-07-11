#!/usr/bin/env python3
"""CAL-05 regression: staleness enforcement must intersect the plugin READ SET.

The bug (CAL-05) -- pscalib's one correctness feature over psana (refuse-by-
default staleness enforcement) fires a FALSE REFUSAL.  ``check_validity`` checked
the validity range of *every* ctype the constants carry and raised
``StaleConstantsError`` if ANY was out of range -- **including ctypes the
detector's apply plugin provably never reads**.

On the demo run (ued1010667/r177, an epix10ka-family detector) the ``geometry``
doc is out of validity range for the run, so the guard refused -- even though
``geometry`` never touches the calibrated output: the epix10ka apply reads only
``pedestals`` / ``pixel_gain`` / ``pixel_status`` / ``mask`` (see
``pscalib.apply.epix10ka`` + ``registry.plugin_epix10ka``), and the assembled
image uses PRE-CACHED pixel-index maps, not the live geometry doc.  psana
returns a finite array, so pscalib's refusal made it strictly WORSE than psana on
the very run it demos.

The fix narrows (does NOT disable) the guard: staleness is enforced only over
``present_ctypes ∩ read_set`` for the detector family.  A stale ctype the plugin
never consumes (``geometry`` for epix10ka) no longer refuses; a stale ctype the
plugin DOES read (``pedestals``) still refuses.

This test is FULLY SELF-CONTAINED: numpy only, NO psana, NO psdata, NO SLAC
data.  It constructs the epix10ka validity scenario directly and drives
``pscalib.model.check_validity``:

  (A) FALSE-REFUSAL (the CAL-05 bug) -- a NON-read ctype (``geometry``) is out of
      range for the run while every READ ctype (``pedestals`` / ``pixel_gain`` /
      ``pixel_status``) is in range.  ``check_validity`` MUST NOT raise (it
      matches psana's finite return).  The family reaches the guard via the
      ``pin`` that already flows in (its ``detname='epixquad'`` names the
      epix10ka family) -- no call-site change -- and, redundantly, via the
      explicit ``family=`` argument.

  (B) GUARD-STILL-FIRES -- a READ ctype (``pedestals``) is out of range.  The
      guard MUST still raise ``StaleConstantsError`` (the fix narrows the check,
      it does not switch it off), and ``geometry`` (still stale) must NOT appear
      among the offenders.

Pre-fix / post-fix discriminator (case A, the load-bearing assertion, run
FIRST):
  * On the PARENT (4b4a2e9f7ff7d000e9bb71cfc855866cfbacef8c, "Merge PR #7:
    fix(cor-03)"), ``check_validity`` checks EVERY ctype, so the stale
    ``geometry`` triggers ``StaleConstantsError`` and the "must not raise"
    assertion FAILS -> the runner exits nonzero.
  * On the FIX, ``check_validity`` intersects with the epix10ka read-set,
    ``geometry`` is excluded, no refusal is raised -> this test passes.  Case B
    still raises for the stale READ ctype, proving the guard is only narrowed.

This file contains NO part of the fix; it is cwd-robust and has a ``main()`` +
``__main__`` entry so ``run_tests.sh`` and a bare ``python3`` both drive it.
"""

import os
import sys

import numpy as np  # noqa: F401  (kept to assert the numpy-only import surface)

# --- locate the pscalib package (parent of this tests dir); cwd-robust -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# The run being calibrated (the epix10ka demo run number is irrelevant to the
# logic; any run works -- what matters is which ctype's range fails to cover it).
RUN = 177


def _epix10ka_validities_geometry_stale():
    """{ctype: Validity} for an epix10ka detector where the NON-read ``geometry``
    doc is OUT of range for ``RUN`` but every READ ctype covers it.

    * pedestals / pixel_gain / pixel_status -- valid ``[100, 'end']`` (cover 177)
    * geometry -- valid ``[10, 50]`` (does NOT cover 177): the CAL-05 offender a
      plugin that never reads geometry must ignore.
    """
    from pscalib.model import Validity
    return {
        "pedestals":    Validity(100, "end"),
        "pixel_gain":   Validity(100, "end"),
        "pixel_status": Validity(100, "end"),
        "geometry":     Validity(10, 50),          # stale for RUN=177
    }


def _epix10ka_pin():
    """A :class:`Pin` naming the epix10ka family exactly as a real snapshot's
    ``pin_obj`` does (``detname='epixquad'`` + an ``epix10ka_...`` uniqueid), so
    the guard recovers the read-set from the pin already threaded through the
    existing call -- no registry / call-site change."""
    from pscalib.model import Pin
    return Pin("epix10ka_serial_0123456789", RUN, detname="epixquad",
               exp="ued1010667")


# ==========================================================================
# (A) FALSE-REFUSAL: a stale NON-read ctype (geometry) must NOT refuse
# ==========================================================================
def test_no_false_refusal_on_nonread_ctype():
    from pscalib.model import check_validity, StaleConstantsError

    validities = _epix10ka_validities_geometry_stale()
    pin = _epix10ka_pin()

    # sanity: the scenario is a GENUINE out-of-range for geometry (so a
    # check-everything guard, i.e. the parent, would refuse) but NOT for any
    # ctype the epix10ka plugin reads.
    assert not validities["geometry"].contains(RUN), (
        "test scenario is wrong: geometry must be out of range for the run")
    for ct in ("pedestals", "pixel_gain", "pixel_status"):
        assert validities[ct].contains(RUN), (
            f"test scenario is wrong: {ct} must be in range for the run")

    # --- the load-bearing discriminator: pin-derived family, SAME call
    # signature as the parent (no new kwarg), so parent-vs-fix differ ONLY in
    # the false-refusal behavior.  Parent: geometry stale -> raises. Fix:
    # geometry not in the epix10ka read-set -> no raise.
    try:
        offenders = check_validity(validities, RUN, pin=pin)
    except StaleConstantsError as e:
        raise AssertionError(
            "CAL-05 FALSE REFUSAL: check_validity raised StaleConstantsError "
            "because the NON-read ctype 'geometry' is stale, even though the "
            "epix10ka plugin never reads geometry (psana returns a finite "
            f"array here).  offenders={e.offenders}")
    assert offenders == [], (
        f"expected no offenders once geometry is excluded; got {offenders}")
    print("[ok] (A) pin-derived epix10ka read-set: stale non-read 'geometry' "
          "does NOT refuse (no false refusal)")

    # --- redundant proof that the explicit family= route narrows identically.
    assert check_validity(validities, RUN, family="epix10ka") == []
    assert check_validity(validities, RUN,
                          family="epix10ka_raw_2_0_1") == []   # versioned class
    print("[ok] (A) explicit family='epix10ka' read-set: same non-refusal")

    # --- allow_stale corroboration: with the guard narrowed, there is nothing
    # to warn about (geometry excluded), so the offender list is EMPTY -- a
    # check-everything guard would return [('geometry', ...)] instead.
    warned = check_validity(validities, RUN, pin=pin, allow_stale=True)
    assert warned == [], (
        f"narrowed guard should have no offenders to warn about; got {warned}")
    print("[ok] (A) allow_stale returns no offenders (geometry excluded)")


# ==========================================================================
# (B) GUARD-STILL-FIRES: a stale READ ctype (pedestals) must still refuse
# ==========================================================================
def test_guard_still_fires_on_read_ctype():
    from pscalib.model import check_validity, StaleConstantsError, Validity

    pin = _epix10ka_pin()
    # pedestals (a READ ctype) is now OUT of range for RUN; geometry is ALSO
    # still stale (a NON-read ctype).  The guard must fire for pedestals and
    # must NOT list geometry among the offenders.
    validities = {
        "pedestals":    Validity(10, 50),          # stale READ ctype
        "pixel_gain":   Validity(100, "end"),      # in range
        "pixel_status": Validity(100, "end"),      # in range
        "geometry":     Validity(10, 50),          # stale NON-read ctype
    }

    raised = False
    try:
        check_validity(validities, RUN, pin=pin)
    except StaleConstantsError as e:
        raised = True
        offending = {ct for ct, _ in e.offenders}
        assert "pedestals" in offending, (
            f"the stale READ ctype 'pedestals' must be an offender; got "
            f"{offending}")
        assert "geometry" not in offending, (
            f"the NON-read ctype 'geometry' must NOT be an offender (it is "
            f"excluded from the epix10ka read-set); got {offending}")
    assert raised, (
        "guard was DISABLED, not narrowed: a stale READ ctype (pedestals) must "
        "still raise StaleConstantsError")
    print("[ok] (B) guard still fires for the stale READ ctype 'pedestals' "
          "(and geometry is not among the offenders) -- guard narrowed, not off")


# ==========================================================================
# (C) BACKWARD COMPAT: unknown/undeterminable family still checks EVERY ctype
# ==========================================================================
def test_unknown_family_checks_all_ctypes():
    """When the family cannot be determined (no family=, no identifying pin),
    the guard falls back to checking every ctype -- the pre-CAL-05 behavior --
    so a stale ctype still refuses.  This pins the SAFE default."""
    from pscalib.model import check_validity, StaleConstantsError

    validities = _epix10ka_validities_geometry_stale()  # only geometry stale
    raised = False
    try:
        check_validity(validities, RUN)                 # no family, no pin
    except StaleConstantsError as e:
        raised = True
        assert "geometry" in {ct for ct, _ in e.offenders}, e.offenders
    assert raised, (
        "with no way to identify the family the guard must check EVERY ctype "
        "(pre-CAL-05 behavior) and refuse on the stale geometry")
    print("[ok] (C) no family / no pin -> checks all ctypes (safe default "
          "preserved), stale geometry still refuses")


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CAL-05 regression: staleness intersects the apply-plugin READ SET")
    print("=" * 72)
    # (A) FIRST: the load-bearing false-refusal discriminator (fails on parent).
    test_no_false_refusal_on_nonread_ctype()
    # (B) the guard is only narrowed, not disabled.
    test_guard_still_fires_on_read_ctype()
    # (C) the safe default (check-all) is preserved when the family is unknown.
    test_unknown_family_checks_all_ctypes()
    print("\nALL CAL-05 REGRESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
