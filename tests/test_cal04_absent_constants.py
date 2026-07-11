#!/usr/bin/env python3
"""CAL-04 regression: an ABSENT optional constant must default (psana-graceful).

The bug (CAL-04) -- *pscalib raises where psana degrades gracefully*: when an
optional calibration constant is missing from the DB/snapshot, psana substitutes
a default and returns a FINITE array -- a missing ``pixel_gain`` defaults to
all-ones (gain factor 1 everywhere -> no gain correction), a missing
``pixel_offset`` to zeros.  pscalib pulled ``pixel_gain`` with ``required=True``,
so an absent ``pixel_gain`` made ``pscalib.calib(...)`` raise ``KeyError`` (from
``registry._get_const``): a run whose DB merely lacks a constant psana tolerates
could not be calibrated at all.  Evidence: EXEC:31300735 (delete ``pixel_gain``;
psana returns a finite array, pscalib raises ``KeyError``).

This test is FULLY SELF-CONTAINED: numpy only, NO psana, NO psdata, NO SLAC
data.  It builds synthetic jungfrau- and epix10ka-shaped raw + constants and:

  (1) ABSENT ``pixel_gain`` (pedestals present) -- ``pscalib.calib`` must return
      a FINITE calibrated array byte-identical to the SAME call with
      ``pixel_gain = ones`` (gain factor 1 -- no gain correction), instead of
      raising.  Checked for BOTH the jungfrau and epix10ka apply plugins (the
      fix touches both).

  (2) ALL-PRESENT (a non-trivial, non-ones ``pixel_gain``) -- ``calib`` must be
      byte-for-byte identical to the pure apply leaf, proving the fix does NOT
      change behavior when the constant IS present (no false behavior change).

  (3) ABSENT ``pedestals`` -- an ESSENTIAL constant psana also cannot do without
      must STILL be refused (with a clear error), never silently fabricated.
      "Do not paper over a truly-missing essential constant."

Pre-fix / post-fix discriminator (case 1):
  * On the PARENT (4b4a2e9f7ff7d000e9bb71cfc855866cfbacef8c, "Merge PR #7:
    fix(cor-03)"), ``registry.plugin_jungfrau`` / ``plugin_epix10ka`` pull
    ``pixel_gain`` with ``_get_const(..., required=True)``, so an absent
    ``pixel_gain`` raises ``KeyError`` -- the "must return finite" assertion
    FAILS and the runner exits nonzero.
  * On the FIX, an absent ``pixel_gain`` is substituted with a ones array shaped
    like pedestals (psana's default), so ``calib`` returns the SAME finite array
    it would for ``pixel_gain = ones`` -> this test passes.

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

# --------------------------------------------------------------------------
# Synthetic jungfrau shapes (3 gain stages; per-segment 512 x 1024).  Segment
# counts are kept tiny to stay light on memory.
# --------------------------------------------------------------------------
JF_ROWS, JF_COLS = 512, 1024
JF_STAGES = 3
JF_NSEG = 2

# Synthetic epix10ka shapes (7 gain ranges; per-segment 352 x 384).
EK_ROWS, EK_COLS = 352, 384
EK_RANGES = 7
EK_NSEG = 4


def _jf_raw(n_seg=JF_NSEG):
    """A synthetic jungfrau raw stack ``(n_seg, 512, 1024)`` uint16, all in gain
    stage 0 (top two bits clear) with distinct ADC codes so the apply produces
    non-trivial, finite output."""
    rng = np.arange(n_seg * JF_ROWS * JF_COLS, dtype=np.uint16) % 97
    return rng.reshape(n_seg, JF_ROWS, JF_COLS)


def _jf_pedestals(n_seg=JF_NSEG):
    """Non-trivial synthetic jungfrau pedestals ``(3, n_seg, 512, 1024)`` f32 --
    nonzero so pedestal subtraction (which survives a missing gain) is visible."""
    return (np.arange(JF_STAGES * n_seg * JF_ROWS * JF_COLS, dtype=np.float32)
            % 13.0).reshape(JF_STAGES, n_seg, JF_ROWS, JF_COLS)


def _ek_raw(n_seg=EK_NSEG):
    """A synthetic epix10ka raw stack ``(n_seg, 352, 384)`` uint16 with ADC codes
    in the low 14 bits (bit 14, the data gain bit, clear) so the gain-range
    decode is stable."""
    rng = np.arange(n_seg * EK_ROWS * EK_COLS, dtype=np.uint16) % 101
    return rng.reshape(n_seg, EK_ROWS, EK_COLS)


def _ek_pedestals(n_seg=EK_NSEG):
    """Non-trivial synthetic epix10ka pedestals ``(7, n_seg, 352, 384)`` f32."""
    return (np.arange(EK_RANGES * n_seg * EK_ROWS * EK_COLS, dtype=np.float32)
            % 7.0).reshape(EK_RANGES, n_seg, EK_ROWS, EK_COLS)


def _ek_config(n_seg=EK_NSEG):
    """A synthetic per-segment epix10ka Configure mapping ``{seg: seg_cfg}`` with
    ``seg_cfg.config.{trbit, asicPixelConfig}`` -- the load-bearing per-ASIC
    config the epix10ka gain-range decode needs (NOT a calib-DB constant)."""
    class _Ns:
        def __init__(self, trbit, apc):
            self.trbit = trbit
            self.asicPixelConfig = apc

    class _Seg:
        def __init__(self, cfg):
            self.config = cfg

    return {i: _Seg(_Ns(np.zeros(4, np.uint8),
                        np.zeros((4, 176, 192), np.uint8)))
            for i in range(n_seg)}


# --------------------------------------------------------------------------
# (1) ABSENT pixel_gain -> finite, == the pixel_gain==ones result (NOT a raise)
# --------------------------------------------------------------------------
def _assert_absent_gain_defaults_to_ones(det_type, raw, pedestals, config=None,
                                         label=""):
    """Core discriminator: with ``pixel_gain`` ABSENT (pedestals present),
    ``pscalib.calib`` must return a finite array byte-identical to the SAME call
    with an explicit ``pixel_gain = ones`` -- not raise ``KeyError``."""
    import pscalib

    # psana's default for a missing pixel_gain: ones (gain factor 1 everywhere).
    ones_gain = np.ones_like(np.asarray(pedestals, dtype=np.float32))
    ref = pscalib.calib(det_type, raw,
                        {"pedestals": pedestals, "pixel_gain": ones_gain},
                        config=config)

    returned = None
    raised = None
    try:
        returned = pscalib.calib(det_type, raw, {"pedestals": pedestals},
                                 config=config)
    except Exception as e:               # noqa: BLE001 -- any raise is the bug
        raised = e

    # THE CAL-04 assertion: an absent optional constant must NOT raise; it must
    # degrade to psana's default and return a finite array.
    assert raised is None, (
        "CAL-04 REGRESSION (%s): pscalib.calib raised %r for an ABSENT "
        "pixel_gain, but psana degrades gracefully (a missing pixel_gain "
        "defaults to ones -> gain factor 1 -> finite output). A run whose DB "
        "merely lacks pixel_gain cannot be calibrated at all." % (
            label, raised))
    assert returned is not None
    assert returned.shape == ref.shape and returned.dtype == np.float32, (
        returned.shape, returned.dtype)
    assert np.isfinite(returned).all(), (
        "CAL-04 (%s): absent-gain apply produced non-finite values" % label)
    # The absent-gain result must be EXACTLY the pixel_gain==ones result: gain
    # factor 1 everywhere (no gain correction), pedestal subtraction preserved.
    assert np.array_equal(returned, ref), (
        "CAL-04 (%s): absent pixel_gain did not match the pixel_gain==ones "
        "result; max|diff|=%s" % (
            label, np.abs(np.nan_to_num(returned) - np.nan_to_num(ref)).max()))
    print("[ok] (1) %s: absent pixel_gain -> finite %r f32, byte-identical to "
          "pixel_gain==ones (gain factor 1, no correction)" % (
              label, tuple(returned.shape)))


def test_jungfrau_absent_pixel_gain_defaults_to_ones():
    _assert_absent_gain_defaults_to_ones(
        "jungfrau", _jf_raw(), _jf_pedestals(), config=None, label="jungfrau")


def test_epix10ka_absent_pixel_gain_defaults_to_ones():
    _assert_absent_gain_defaults_to_ones(
        "epix10ka", _ek_raw(), _ek_pedestals(), config=_ek_config(),
        label="epix10ka")


# --------------------------------------------------------------------------
# (2) ALL-PRESENT (non-ones gain) -> byte-identical to the leaf (UNCHANGED)
# --------------------------------------------------------------------------
def test_all_present_byte_unchanged():
    """With every constant PRESENT and a non-trivial (non-ones) ``pixel_gain``,
    the guarded ``pscalib.calib`` dispatch must be byte-for-byte identical to the
    pure apply leaf.  The fix only adds a branch for the ABSENT case, so the
    present path is unchanged -- this passes on parent AND fix, guarding against
    a false behavior change."""
    import pscalib

    # jungfrau: non-ones gain so a wrong (e.g. accidental ones) substitution
    # would show up as a byte difference vs the leaf.
    jf_raw = _jf_raw()
    jf_ped = _jf_pedestals()
    jf_gain = np.full((JF_STAGES, JF_NSEG, JF_ROWS, JF_COLS), 2.0, np.float32)
    jf_out = pscalib.calib("jungfrau", jf_raw,
                           {"pedestals": jf_ped, "pixel_gain": jf_gain})
    jf_ref = pscalib.calib_jungfrau(jf_raw, jf_ped, jf_gain)
    assert np.array_equal(jf_out, jf_ref), (
        "CAL-04: jungfrau all-present output changed vs the apply leaf "
        "(present-constant behavior must be byte-unchanged)")

    # epix10ka: non-ones gain, all present.
    ek_raw = _ek_raw()
    ek_ped = _ek_pedestals()
    ek_cfg = _ek_config()
    ek_gain = np.full((EK_RANGES, EK_NSEG, EK_ROWS, EK_COLS), 3.0, np.float32)
    ek_out = pscalib.calib("epix10ka", ek_raw,
                           {"pedestals": ek_ped, "pixel_gain": ek_gain},
                           config=ek_cfg)
    ek_ref = pscalib.calib_epix10ka(ek_raw, ek_ped, ek_gain, ek_cfg)
    assert np.array_equal(ek_out, ek_ref), (
        "CAL-04: epix10ka all-present output changed vs the apply leaf "
        "(present-constant behavior must be byte-unchanged)")
    print("[ok] (2) all-present (non-ones gain): jungfrau + epix10ka dispatch "
          "byte-identical to the apply leaf (no false behavior change)")


# --------------------------------------------------------------------------
# (3) ABSENT pedestals -> still refused (do not paper over an essential const)
# --------------------------------------------------------------------------
def test_absent_pedestals_still_refused():
    """A genuinely REQUIRED constant (``pedestals``) that psana also cannot do
    without must STILL fail -- never a silently-fabricated default.  (This holds
    on both parent and fix; it guards the fix from over-reaching into essential
    constants.)  The refusal must be clear: it names the missing ctype."""
    import pscalib

    raw = _jf_raw()
    gain = np.ones((JF_STAGES, JF_NSEG, JF_ROWS, JF_COLS), np.float32)

    raised = None
    try:
        pscalib.calib("jungfrau", raw, {"pixel_gain": gain})   # NO pedestals
    except Exception as e:               # noqa: BLE001
        raised = e
    assert raised is not None, (
        "CAL-04: a MISSING pedestals (an essential constant psana also requires) "
        "must be refused, not silently defaulted")
    assert "pedestals" in str(raised), (
        "CAL-04: the missing-pedestals refusal must name the essential ctype; "
        "got: %r" % (raised,))
    print("[ok] (3) absent pedestals: still refused with a clear error naming "
          "the essential ctype (%s)" % type(raised).__name__)


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CAL-04 regression: an absent optional constant defaults (psana-"
          "graceful), an essential one is still refused")
    print("=" * 72)
    test_jungfrau_absent_pixel_gain_defaults_to_ones()
    test_epix10ka_absent_pixel_gain_defaults_to_ones()
    test_all_present_byte_unchanged()
    test_absent_pedestals_still_refused()
    print("\nALL CAL-04 CHECKS PASSED (absent pixel_gain -> psana's ones "
          "default; present constants byte-unchanged; absent pedestals refused)")


if __name__ == "__main__":
    main()
