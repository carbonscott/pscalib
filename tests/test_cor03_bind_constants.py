#!/usr/bin/env python3
"""COR-03 regression: pscalib.calib must BIND the constants to the raw.

The bug (COR-03) -- *nothing binds constants to the raw they are applied to*:
``pscalib.calib(det_type, raw, constants)`` dispatches to the family apply
plugin and applies ``constants`` (pedestals / pixel_gain / ...) to ``raw`` with
NO check that the constants actually correspond to that raw.  The apply leaf
loops the raw's segments and indexes the constants per segment, so same-family
constants of a DIFFERENT detector -- a different panel count, a different
per-segment pixel geometry, or a different jungfrau entirely -- are applied
silently and yield a finite but wholly WRONG calibrated image, no error.  A
scientist can calibrate detector A's data with detector B's pedestals/gain and
publish a plausible-looking, completely wrong result.

This test is FULLY SELF-CONTAINED: numpy only, NO psana, NO psdata, NO SLAC
data.  It builds a synthetic jungfrau-shaped raw and:

  (1) MATCHING constants (segment count + per-segment geometry corresponding to
      the raw) -- ``calib`` must return a finite calibrated array with NO false
      alarm, byte-identical to the raw apply leaf (correct usage is UNCHANGED).

  (2) NON-CORRESPONDING constants (a DIFFERENT segment count) -- ``calib`` must
      RAISE a clear error naming raw shape vs constants shape, instead of
      silently returning a finite image.

Pre-fix / post-fix discriminator (case 2):
  * On the PARENT (5133733544d1b927b454df5fd9b22d91cd605869, "Merge PR #5:
    fix(cal-15)"), the mismatch is SILENTLY tolerated -- the jungfrau apply's
    per-segment loop indexes the leading segments of the larger constants stack
    -- so ``calib`` returns a finite ``(N,512,1024)`` image and the "must raise"
    assertion FAILS -> the runner exits nonzero.
  * On the FIX, ``calib`` binds the constants to the raw before dispatch and
    raises ``pscalib.registry.ConstantsRawMismatchError`` -> this test passes.

Residual gap (documented, not tested here because it is NOT catchable at this
seam): two DIFFERENT detectors of the SAME family and SAME shape (jungfrau A vs
jungfrau B, both ``(32,512,1024)``) are indistinguishable by shape alone -- the
raw is a bare ndarray carrying no detector identity to bind the constants'
provenance against.  Shape binding closes the wrong-panel-count / wrong-geometry
class; same-shape-different-detector needs identity on the raw side too.

This file contains NO part of the fix; it is cwd-robust and has a ``main()`` +
``__main__`` entry so ``run_tests.sh`` and a bare ``python3`` both drive it.
"""

import os
import sys

import numpy as np

# --- locate the pscalib package (parent of this tests dir); cwd-robust -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# A small but faithfully jungfrau-shaped per-segment geometry (512 x 1024).
# Segment counts are kept tiny to stay light on memory; only the segment-count
# CORRESPONDENCE (not the absolute count) is what this test exercises.
ROWS, COLS = 512, 1024
N_STAGES = 3                       # jungfrau: leading axis = 3 gain stages
N_SEG = 2                          # the raw's segment count this "event"
M_SEG = 4                          # a DIFFERENT detector's segment count (M != N)


def _jf_raw(n_seg=N_SEG):
    """A synthetic jungfrau raw stack ``(n_seg, 512, 1024)`` uint16.

    Values are kept in gain stage 0 (top two bits clear) with a few distinct
    ADC codes so the matching apply produces non-trivial, finite output.
    """
    rng = np.arange(n_seg * ROWS * COLS, dtype=np.uint16) % 97
    return rng.reshape(n_seg, ROWS, COLS)


def _jf_constants(n_seg):
    """Well-formed synthetic jungfrau constants for ``n_seg`` segments:
    ``pedestals`` / ``pixel_gain`` shaped ``(3, n_seg, 512, 1024)`` (nonzero
    gain so the divide is finite)."""
    ped = np.zeros((N_STAGES, n_seg, ROWS, COLS), np.float32)
    gain = np.ones((N_STAGES, n_seg, ROWS, COLS), np.float32)
    return {"pedestals": ped, "pixel_gain": gain}


# --------------------------------------------------------------------------
# (1) MATCHING constants -> no false alarm, output byte-identical to the leaf
# --------------------------------------------------------------------------
def test_matching_constants_no_false_alarm():
    """Constants whose segment set + geometry CORRESPOND to the raw must
    dispatch exactly as before -- a finite calibrated array, byte-identical to
    the pure apply leaf (which bypasses the binding gate).  This is the
    byte-unchanged guarantee for correct usage."""
    import pscalib

    raw = _jf_raw(N_SEG)
    cons = _jf_constants(N_SEG)

    out = pscalib.calib("jungfrau", raw, cons)
    assert out.shape == (N_SEG, ROWS, COLS), out.shape
    assert out.dtype == np.float32, out.dtype
    assert np.isfinite(out).all(), "matching apply produced non-finite values"

    # The leaf apply takes no binding gate -> the guarded dispatch must be
    # byte-for-byte identical to it for corresponding constants.
    ref = pscalib.calib_jungfrau(raw, cons["pedestals"], cons["pixel_gain"])
    assert np.array_equal(out, ref), (
        "COR-03 binding changed the output for CORRESPONDING constants "
        "(correct usage must be byte-unchanged)")
    print("[ok] (1) matching constants: finite (%d,%d,%d) f32, byte-identical "
          "to the apply leaf (no false alarm)" % (N_SEG, ROWS, COLS))


# --------------------------------------------------------------------------
# (2) NON-CORRESPONDING constants (different segment count) -> must RAISE
# --------------------------------------------------------------------------
def test_segment_count_mismatch_raises():
    """A DIFFERENT detector's constants (more segments than the raw) must be
    REFUSED with a clear error, not silently applied.

    This is the pre-fix / post-fix discriminator: the parent's jungfrau apply
    loops ``for s in range(raw.shape[0])`` and indexes the LEADING segments of
    the larger constants stack, so it silently returns a finite ``(N,512,1024)``
    image.  The fix binds the constants to the raw first and raises.
    """
    import pscalib

    raw = _jf_raw(N_SEG)                 # N = 2 segments this event
    mismatched = _jf_constants(M_SEG)    # constants for M = 4 segments (M != N)

    returned = None
    raised = None
    try:
        returned = pscalib.calib("jungfrau", raw, mismatched)
    except Exception as e:               # noqa: BLE001 -- any refusal counts here
        raised = e

    # THE COR-03 assertion: a non-corresponding set must NOT silently calibrate.
    assert raised is not None, (
        "COR-03 REGRESSION: pscalib.calib SILENTLY applied constants for a "
        "%d-segment detector to a %d-segment raw and returned a finite %r image "
        "instead of refusing. Nothing bound the constants to the raw they were "
        "applied to -- detector A's data was calibrated with detector B's "
        "constants with no error." % (
            M_SEG, N_SEG,
            None if returned is None else tuple(returned.shape)))

    # The refusal must be an argument-value error (ValueError family) whose
    # message names BOTH shapes -- a clear, actionable error, not an opaque one.
    assert isinstance(raised, ValueError), (
        "COR-03 refusal must subclass ValueError; got %r" % (raised,))
    msg = str(raised)
    assert str((N_SEG, ROWS, COLS)) in msg and str((N_STAGES, M_SEG, ROWS, COLS)) in msg, (
        "COR-03 refusal must name the raw shape AND the constants shape; got: "
        + msg)

    # And it is the dedicated, typed COR-03 signal (not some unrelated error).
    import pscalib.registry as reg
    exc_cls = getattr(reg, "ConstantsRawMismatchError", None)
    assert exc_cls is not None and isinstance(raised, exc_cls), (
        "COR-03 refusal must be the dedicated ConstantsRawMismatchError; "
        "got %r" % (raised,))
    assert raised.raw_segments == N_SEG and raised.const_segments == M_SEG
    assert raised.ctype in ("pedestals", "pixel_gain")
    print("[ok] (2) segment-count mismatch (raw N=%d vs constants M=%d) RAISED "
          "%s naming both shapes" % (N_SEG, M_SEG, type(raised).__name__))


# --------------------------------------------------------------------------
# (3) NON-CORRESPONDING constants (different pixel geometry) -> must RAISE
# --------------------------------------------------------------------------
def test_pixel_geometry_mismatch_raises():
    """Constants with a different per-segment pixel geometry (same segment
    count) must also be refused -- a wrong-geometry stack would broadcast wrong
    or blow up deep in numpy; the fix refuses first with a clear error."""
    import pscalib
    import pscalib.registry as reg

    raw = _jf_raw(N_SEG)                                  # (2, 512, 1024)
    bad_geo = {"pedestals": np.zeros((N_STAGES, N_SEG, 256, 256), np.float32),
               "pixel_gain": np.ones((N_STAGES, N_SEG, 256, 256), np.float32)}

    exc_cls = getattr(reg, "ConstantsRawMismatchError", None)
    raised = None
    try:
        pscalib.calib("jungfrau", raw, bad_geo)
    except Exception as e:               # noqa: BLE001
        raised = e
    assert raised is not None, (
        "COR-03: constants with a wrong per-segment pixel geometry "
        "(256x256 vs the raw's 512x1024) were not refused")
    assert exc_cls is not None and isinstance(raised, exc_cls), (
        "pixel-geometry mismatch must raise ConstantsRawMismatchError; got %r"
        % (raised,))
    assert "(256, 256)" in str(raised) and "(512, 1024)" in str(raised), str(raised)
    print("[ok] (3) pixel-geometry mismatch (256x256 vs 512x1024) RAISED %s"
          % type(raised).__name__)


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("COR-03 regression: pscalib.calib must bind constants to the raw")
    print("=" * 72)
    test_matching_constants_no_false_alarm()
    test_segment_count_mismatch_raises()
    test_pixel_geometry_mismatch_raises()
    print("\nALL COR-03 CHECKS PASSED (constants are bound to the raw before "
          "apply; corresponding constants are byte-unchanged)")


if __name__ == "__main__":
    main()
