"""CAL-11 regression: ``mask_from_pixel_status`` must default to ``status_bits``
== ``0xffff`` (psana's ``UtilsMask.status_as_mask`` default), i.e. only the LOW
16 status bits decide masking.

Self-contained: numpy only -- NO psana, NO psdata, NO SLAC data.  Runs anywhere.
It is the pre-fix / post-fix discriminator for CAL-11 and contains NO part of the
fix itself.

The bug (CAL-11): pscalib's ``mask_from_pixel_status`` defaulted to
``status_bits=(1<<64)-1`` (all 64 bits), while psana masks only on the low 16
bits (``0xffff``).  Undetectable on real data today because no reachable pixel
carries a status bit above bit 15 -- but a pixel with a status bit >= 16 (which
psana IGNORES) would be masked by the old default, diverging from
``det.raw.calib(evt)``.

The probe below builds a synthetic ``pixel_status`` with a pixel whose status
word sets bit 20 and is zero in the low 16 bits.  psana keeps that pixel;
pscalib's OLD default masks it.

  * On the PARENT (default ``(1<<64)-1``): that pixel is masked (mask == 0) ->
    the KEEP assertion fails -> exit 1.
  * On the FIX     (default ``0xffff``)  : that pixel is kept   (mask == 1) ->
    all assertions pass -> exit 0.
"""

import os
import sys

import numpy as np

# Robust to cwd: derive the package src/ dir from THIS file's location
# (.../pscalib/tests/test_cal11_status_bits.py -> .../pscalib/src).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pscalib.apply.epix10ka import mask_from_pixel_status  # noqa: E402

# A status word with ONLY bit 20 set: >= bit 16, and all-zero in the low 16
# bits.  psana (0xffff) ignores it -> pixel kept; the old pscalib default
# (all 64 bits) honours it -> pixel masked.
_HIGH_BIT = 20
_HIGH_ONLY = np.uint64(1) << np.uint64(_HIGH_BIT)   # 0x100000 == 1<<20
_LOW_BIT = np.uint64(1)                             # bit 0 -- a "normal" bad bit
_WIDE = (1 << 64) - 1                               # the OLD (buggy) default

# Positions in the final (n_seg, H, W) mask.
_SEG, _H, _W = 1, 3, 3
_A = (0, 0, 0)   # high-bit-only pixel  -> psana KEEPS (mask 1)
_B = (0, 0, 1)   # low-bit pixel        -> masked (mask 0) under ANY sane default
_C = (0, 0, 2)   # clean pixel (0)      -> kept (mask 1)


def _make_status():
    """Synthetic ``pixel_status`` (7 gain ranges, 1 seg, 3x3), uint64.

    The three probe pixels carry the SAME status word across all 7 gain-range
    planes so the gain-range AND-merge (:func:`merge_mask_for_grinds`, ranges
    0-4) is deterministic and reflects each pixel's status bits directly.
    """
    base = np.zeros((_SEG, _H, _W), dtype=np.uint64)
    base[_A] = _HIGH_ONLY      # bit 20 only
    base[_B] = _LOW_BIT        # bit 0
    # _C stays 0 (clean)
    # replicate the same plane across all 7 gain ranges
    status = np.broadcast_to(base, (7, _SEG, _H, _W)).copy()
    assert status.shape == (7, _SEG, _H, _W)
    # sanity: the high-bit pixel really is zero in the low 16 bits
    assert int(status[(0,) + _A]) & 0xffff == 0
    assert int(status[(0,) + _A]) >> 16 != 0
    return status


def test_default_ignores_high_status_bits():
    """DEFAULT ``mask_from_pixel_status`` keeps a pixel whose only status bit is
    >= 16 (matches psana's ``0xffff``).  FAILS on the parent, PASSES on the fix.
    """
    status = _make_status()
    mask = mask_from_pixel_status(status)          # DEFAULT status_bits
    assert mask.shape == (_SEG, _H, _W), mask.shape

    # THE DISCRIMINATOR: high-bit-only pixel must be KEPT under the default.
    # Parent default (1<<64)-1 masks it (mask 0) -> this assertion fails.
    assert int(mask[_A]) == 1, (
        "CAL-11: default mask_from_pixel_status masked a pixel whose only "
        "status bit is bit %d (>= 16); psana (status_bits=0xffff) keeps it. "
        "Default must be 0xffff, not a wide 64-bit mask. got mask[A]=%d"
        % (_HIGH_BIT, int(mask[_A])))

    # Normal masking must be unchanged: a low-bit (bit 0) pixel is still masked,
    # a clean pixel is still kept.
    assert int(mask[_B]) == 0, "low-bit (bit 0) pixel must be masked under default"
    assert int(mask[_C]) == 1, "clean (status 0) pixel must be kept under default"


def test_explicit_status_bits_still_selectable():
    """A caller may still pass an explicit ``status_bits``; narrow (0xffff) and
    wide ((1<<64)-1) DIFFER precisely on the high-bit pixel -- proving the fix
    changed only the DEFAULT, not the parameter's meaning.
    """
    status = _make_status()

    mask_narrow = mask_from_pixel_status(status, status_bits=0xffff)
    mask_wide = mask_from_pixel_status(status, status_bits=_WIDE)

    # High-bit pixel: kept under 0xffff, masked under the wide 64-bit mask.
    assert int(mask_narrow[_A]) == 1, "0xffff must KEEP the bit-20-only pixel"
    assert int(mask_wide[_A]) == 0, "wide 64-bit mask must MASK the bit-20 pixel"
    assert int(mask_narrow[_A]) != int(mask_wide[_A]), (
        "narrow (0xffff) and wide (1<<64-1) must differ on the high-bit pixel")

    # An explicit mask targeting bit 20 also masks it (parameter is honoured).
    mask_bit20 = mask_from_pixel_status(status, status_bits=(1 << _HIGH_BIT))
    assert int(mask_bit20[_A]) == 0, "explicit status_bits=1<<20 must mask it"

    # Normal (low-bit) masking is identical under BOTH narrow and wide: bit 0 is
    # within the low 16, so it is masked either way; clean stays kept either way.
    for m, name in ((mask_narrow, "0xffff"), (mask_wide, "wide")):
        assert int(m[_B]) == 0, "bit-0 pixel must be masked under %s" % name
        assert int(m[_C]) == 1, "clean pixel must be kept under %s" % name


def main():
    tests = (
        test_default_ignores_high_status_bits,
        test_explicit_status_bits_still_selectable,
    )
    for t in tests:
        t()
        print("[PASS] %s" % t.__name__)
    print("CAL-11 regression: all assertions passed "
          "(mask_from_pixel_status default == 0xffff, psana-faithful).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
