#!/usr/bin/env python3
"""CAL-12 regression: the staleness guard's FORWARD-expiry direction.

Parent SHA: 884cc20fd7f031d1c588a876ea9813d88c17bce5
  (origin/main -- merged CAL-05 read-set intersection + all prior pscalib fixes)

Background -- the CAL-12 defect matrix
--------------------------------------
The US-002 staleness guard (:func:`pscalib.model.check_validity`) is meant to
refuse constants whose validity range does NOT cover the requested run, in BOTH
directions:

  * BACKWARD -- ``run < run`` (first-valid): too early, the constant did not
    exist yet.
  * FORWARD  -- ``run > run_end`` (a *finite* last-valid): expired, a newer
    constant superseded it.

CAL-12's observation: on the only real dataset (jungfrau r51) every ctype is
open-ended (``run_end='end'``), so the guard could never fire FORWARD there; the
one forward firing seen on real data was CAL-05's false positive (a non-read
``geometry`` doc). So the forward-expiry path had never been exercised
*correctly* -- and this test does exactly that, offline, with numpy only.

What this file proves (all offline, no psana, no SLAC data)
-----------------------------------------------------------
1.  FORWARD FIRES -- a READ ctype (``pedestals``) with a FINITE ``run_end``
    BELOW the requested run raises ``StaleConstantsError`` naming it, while the
    other (valid) read ctypes do not offend.
2.  CAL-05 PRESERVED -- a forward-expired NON-read ctype (``geometry``) does NOT
    trigger a refusal (it is outside the plugin read-set).
3.  BOUNDARY -- ``run == run_end`` is INCLUSIVE (still valid); expiry fires only
    at ``run == run_end + 1`` and beyond.  Matches psana's vendored
    ``select_doc_in_run_range`` (``begin <= rnum <= end``).
4.  OPEN-ENDED NEVER EXPIRES FORWARD -- the DISCRIMINATOR.  An open-ended
    (``run_end='end'``) READ ctype covers EVERY run at or after its first-valid
    run, including runs beyond ``RUN_MAX`` (9999).  On the parent this FAILS:
    ``Validity.contains`` compares against the capped ``run_end == 9999`` and
    ignores the stored ``open_ended`` flag, so an open-ended constant is falsely
    reported expired for ``run > 9999`` and the guard wrongly raises.  The fix
    honors ``open_ended`` (the ``'end'`` sentinel is unbounded, not the number
    9999).
5.  BACKWARD PRESERVED -- ``run < first-valid`` still refuses for a read ctype.

The DISCRIMINATING assertion is #4 (``test_open_ended_never_expires_forward`` /
the ``Validity.contains`` open-ended check): it RAISES on the parent and PASSES
on the fix.  The other checks pass on both parent and fix -- they lock in the
forward-expiry behavior the matrix says was never tested.
"""

import os
import sys

import numpy as np

# --- locate the pscalib package (parent of this tests dir); cwd-robust --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pscalib.model import (  # noqa: E402
    Validity,
    Pin,
    StaleConstantsError,
    check_validity,
    validities_from_calibconst,
)

# A jungfrau pin -- its read-set (CAL-05) is
# {pedestals, pixel_gain, pixel_offset, pixel_status, mask}; NOT geometry.
_JF_PIN = Pin("jungfrau_serialabc", 51, detname="jungfrau", exp="synthexp")

# psana CalibDoc.rnum_max -- the numeric cap the 'end' sentinel maps to.
_RUN_MAX = Validity.RUN_MAX  # 9999


def _valid_read_ctypes(run_end_for_pedestals):
    """A realistic jungfrau ``{ctype: (ndarray, meta)}`` calibconst dict.

    ``pedestals`` gets the caller's ``run_end`` (finite or ``'end'``); the other
    read ctypes are open-ended and valid, so ``pedestals`` is the only variable.
    """
    def meta(run_end):
        return {"run": 50, "run_end": run_end, "dettype": "jungfrau",
                "version": "v1"}
    small = np.zeros((1, 4, 4), dtype=np.float32)
    return {
        "pedestals":    (small, meta(run_end_for_pedestals)),
        "pixel_gain":   (np.ones((1, 4, 4), np.float32), meta("end")),
        "pixel_status": (np.zeros((1, 4, 4), np.uint16), meta("end")),
        "mask":         (np.ones((1, 4, 4), np.uint8), meta("end")),
    }


def _fires(validities, run, pin=_JF_PIN, family=None):
    """Return the set of offending ctypes if the guard raises, else ``None``."""
    try:
        check_validity(validities, run, allow_stale=False, pin=pin,
                       family=family)
        return None
    except StaleConstantsError as e:
        return {ct for ct, _ in e.offenders}


# ==========================================================================
# (1) FORWARD FIRES: a READ ctype with a FINITE run_end below the run
# ==========================================================================
def test_forward_expiry_fires_on_read_ctype():
    # pedestals valid for [50, 200]; every other read ctype open-ended/valid.
    cc = _valid_read_ctypes(run_end_for_pedestals=200)
    vs = validities_from_calibconst(cc)
    assert vs["pedestals"] == Validity(50, 200), vs
    assert vs["pixel_gain"].open_ended, vs

    # request run 5000 -- well past pedestals' run_end (200), still <= RUN_MAX
    off = _fires(vs, 5000)
    assert off is not None, (
        "FORWARD-expiry guard did NOT fire: a READ ctype (pedestals) whose "
        "finite run_end=200 is below run=5000 must raise StaleConstantsError")
    assert off == {"pedestals"}, (
        "forward-expiry offenders must be exactly the expired READ ctype "
        "(pedestals); the open-ended read ctypes must not offend -- got %r"
        % (off,))

    # and one run past the boundary also fires (run_end + 1)
    assert _fires(vs, 201) == {"pedestals"}, "expiry must fire at run_end+1"
    print("[ok] (1) FORWARD expiry fires: read ctype pedestals(run_end=200) "
          "raises StaleConstantsError for run>200, naming only pedestals")


# ==========================================================================
# (2) CAL-05 PRESERVED: a forward-expired NON-read ctype does NOT refuse
# ==========================================================================
def test_cal05_nonread_forward_expired_does_not_refuse():
    # geometry (NOT in jungfrau's read-set) is forward-expired; every READ
    # ctype covers the run.  CAL-05: the guard must NOT refuse.
    cc = _valid_read_ctypes(run_end_for_pedestals="end")
    cc["geometry"] = (np.zeros((1, 4, 4), np.float32),
                      {"run": 50, "run_end": 200, "dettype": "jungfrau"})
    vs = validities_from_calibconst(cc)
    assert vs["geometry"] == Validity(50, 200), vs

    off = _fires(vs, 5000)          # geometry expired at 200, read ctypes cover
    assert off is None, (
        "CAL-05 regression: a forward-expired NON-read ctype (geometry) must "
        "NOT trigger a refusal -- the guard is intersected with the plugin "
        "read-set; got offenders %r" % (off,))

    # Sanity: WITHOUT the read-set intersection (no family/pin) geometry WOULD
    # offend -- proving CAL-05 is what suppresses it here, not a valid range.
    off_all = _fires(vs, 5000, pin=None)
    assert off_all == {"geometry"}, (
        "check-all mode should see geometry as the sole forward-expired ctype "
        "(confirms CAL-05's read-set intersection is doing the suppression); "
        "got %r" % (off_all,))
    print("[ok] (2) CAL-05 preserved: forward-expired NON-read geometry does "
          "not refuse (read-set intersection intact)")


# ==========================================================================
# (3) BOUNDARY: run == run_end is INCLUSIVE (valid); expiry at run_end+1
# ==========================================================================
def test_boundary_run_equals_run_end_is_inclusive():
    v = Validity(50, 200)
    # chosen semantics: INCLUSIVE at run_end, matching psana's vendored
    # select_doc_in_run_range (begin <= rnum <= end).
    assert v.contains(200) is True, "run == run_end must be VALID (inclusive)"
    assert v.contains(201) is False, "run == run_end+1 must be EXPIRED"

    cc = _valid_read_ctypes(run_end_for_pedestals=200)
    vs = validities_from_calibconst(cc)
    assert _fires(vs, 200) is None, "run == run_end must not refuse (inclusive)"
    assert _fires(vs, 201) == {"pedestals"}, "run_end+1 must refuse"
    print("[ok] (3) boundary INCLUSIVE at run==run_end (psana convention: "
          "begin<=rnum<=end); expiry fires at run_end+1")


# ==========================================================================
# (4) DISCRIMINATOR: open-ended ('end') NEVER expires forward, even huge runs
# ==========================================================================
def test_open_ended_never_expires_forward():
    """An open-ended constant covers every run at/after its first-valid run.

    This is the CAL-12 discriminator.  On the PARENT, ``Validity.contains``
    compares against the capped ``run_end == RUN_MAX == 9999`` and ignores the
    stored ``open_ended`` flag, so it returns False for ``run > 9999`` and the
    guard wrongly raises ``StaleConstantsError`` for a never-expiring constant.
    On the FIX it returns True and the guard stays silent.
    """
    vo = Validity(50, "end")
    assert vo.open_ended and vo.run_end == _RUN_MAX, vo
    # the crisp unit-level discriminator: contains() must honor open_ended.
    assert vo.contains(_RUN_MAX) is True, "open-ended must cover RUN_MAX"
    for huge in (_RUN_MAX + 1, 10000, 100000, 10 ** 9):
        assert vo.contains(huge) is True, (
            "CAL-12: an open-ended ('end') Validity must NEVER expire forward "
            "-- contains(%d) must be True, but the parent (which compares "
            "against the capped run_end=%d and ignores open_ended) returns "
            "False here" % (huge, _RUN_MAX))

    # and through the full guard: an open-ended READ ctype (pedestals) must not
    # refuse for a run far beyond RUN_MAX.
    cc = _valid_read_ctypes(run_end_for_pedestals="end")
    vs = validities_from_calibconst(cc)
    for huge in (_RUN_MAX + 1, 50000):
        off = _fires(vs, huge)
        assert off is None, (
            "CAL-12: open-ended constants must not be refused for a huge run "
            "(%d); the parent falsely fires the forward-expiry guard here. "
            "Offenders: %r" % (huge, off))
    print("[ok] (4) DISCRIMINATOR: open-ended ('end') never expires forward -- "
          "covers RUN_MAX+1, 10000, 100000, 1e9 (parent wrongly refuses these)")


# ==========================================================================
# (5) BACKWARD PRESERVED: run < first-valid still refuses for a read ctype
# ==========================================================================
def test_backward_below_first_valid_still_refuses():
    cc = _valid_read_ctypes(run_end_for_pedestals="end")  # pedestals [50, end]
    vs = validities_from_calibconst(cc)
    # run 0 is below every ctype's first-valid run (50) -> all read ctypes offend
    off = _fires(vs, 0)
    assert off is not None and "pedestals" in off, (
        "BACKWARD direction regressed: run=0 (below first-valid run 50) must "
        "still refuse a read ctype; got %r" % (off,))
    assert off == set(vs), (
        "every read ctype is below its first-valid run at run=0; all must "
        "offend -- got %r vs %r" % (off, set(vs)))
    print("[ok] (5) BACKWARD preserved: run < first-valid still refuses "
          "(run=0 offends every read ctype)")


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CAL-12: forward-expiry direction of the staleness guard "
          "(offline, numpy-only)")
    print("=" * 72)
    test_forward_expiry_fires_on_read_ctype()
    test_cal05_nonread_forward_expired_does_not_refuse()
    test_boundary_run_equals_run_end_is_inclusive()
    test_open_ended_never_expires_forward()
    test_backward_below_first_valid_still_refuses()
    print("\nALL CAL-12 FORWARD-EXPIRY CHECKS PASSED")


if __name__ == "__main__":
    main()
