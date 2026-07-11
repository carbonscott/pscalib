#!/usr/bin/env python3
"""COR-01b regression test: full 7-gain-range census for epix10ka calib.

WHY THIS TEST EXISTS (the COR-01b coverage gap)
-----------------------------------------------
epix10ka calib decodes a per-pixel GAIN RANGE from a 6-bit control word (the
per-ASIC ``trbit`` / ``asicPixelConfig`` config bits OR-ed with the per-event
data gain bit) and applies range-specific ``pedestals`` / ``pixel_gain`` planes
selected from the leading ``(7, ...)`` axis of the constants.  There are SEVEN
gain ranges::

    FH  FM  FL  AHL_H  AML_M  AHL_L  AML_L      (indices 0..6)

On the ONLY real reference dataset (exp=ued1010667, run=177, det='epixquad')
EVERY sampled pixel sits in gain range **FM**.  The US-004 byte-exact proof and
the US-008 mixed-trbit gate therefore exercised the config/quadrant plumbing but
left 6 of the 7 gain-range decode branches (FH, FL, AHL_H, AML_M, AHL_L, AML_L)
never selected by any test.  The calibration was proven for one of seven paths.

WHAT THIS TEST DOES
-------------------
It builds a fully SYNTHETIC epix10ka detector (numpy only; no psana, no psdata,
no SLAC data) whose pixels are laid out so that ALL SEVEN gain ranges are
selected, then drives the real ``pscalib.apply.epix10ka.calib_epix10ka`` and
compares its output, pixel for pixel, against an expected image derived BY HAND
from the documented per-range formula::

    calib[px] = (raw_adc[px] - pedestals[range(px)][px]) * (1 / pixel_gain[range(px)][px])

where ``range(px)`` is decoded from the control word by the psana bit rule
(``psana.detector.UtilsEpix10ka.gain_maps_epix10ka_any_alg``), reproduced here
INDEPENDENTLY (not by calling pscalib's own decode):

    cbits_m60 = cbits & 60   # 0b111100      bits: [data gain][trbit][cfg1][cfg0]
    cbits_m28 = cbits & 28   # 0b011100
    cbits_m12 = cbits & 12   # 0b001100
      FH    = (cbits_m28 == 28)     # 0b011100  cfg1 cfg0 trbit set
      FM    = (cbits_m28 == 12)     # 0b001100  cfg1 cfg0 set, trbit clear
      FL    = (cbits_m12 ==  8)     # 0b001000  cfg1 set, cfg0 clear
      AHL_H = (cbits_m60 == 16)     # 0b010000  trbit set, data clear
      AML_M = (cbits_m60 ==  0)     # 0b000000  all clear
      AHL_L = (cbits_m60 == 48)     # 0b110000  trbit set, data set
      AML_L = (cbits_m60 == 32)     # 0b100000  trbit clear, data set

The seven boolean masks are consumed by an ``np.select`` in ``GAIN_MODES``
order, first match wins.  The canonical control word for each range (used to
lay out the synthetic panel) is::

    FH=28  FM=12  FL=8  AHL_H=16  AML_M=0  AHL_L=48  AML_L=32

FINDING (case A -- coverage gap, NO code defect)
------------------------------------------------
An exhaustive enumeration of all 64 possible 6-bit control words shows pscalib's
``gain_maps_epix10ka_any`` decode is byte-identical to the psana bit rule above
for EVERY word, and each of the 7 range planes is correctly indexed by
``event_constants_for_gmaps``.  So COR-01b is a COVERAGE gap, not a latent bug:
this test PASSES on both the parent tree and any fix (there is no source
change).  Its proof is "the all-7-gain-ranges census runs green", closing the
6-of-7 untested-branch gap -- NOT a parent-vs-fix delta.

The test is self-validating: it asserts all 7 ranges are genuinely populated
(a run that silently collapsed to FM would fail the census), that the 7 ranges
partition the panel, that pscalib's decode masks equal the independent psana
masks per range, and -- as a negative control -- that mis-assigning any range's
constant plane WOULD change the output (so the byte-exact match is load-bearing,
not vacuous).

Branched off origin/main @ 6d7870a754b68c991dc1df484d2543ac4cf856ca.

Runs entirely offline: ``python3 tests/test_cor01b_gain_census.py`` (or via
pytest).  No psana / psdata / SLAC data required.
"""

import os
import sys

import numpy as np

# --- locate the pscalib package (parent of this tests dir), cwd-robust ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pscalib.apply import epix10ka as e  # noqa: E402

# Panel geometry (epix10ka: a 352x384 panel of four 176x192 ASICs).
R, C = e.PANEL_SHAPE            # 352, 384
RH, CH = R // 2, C // 2        # 176, 192  (one ASIC / quadrant)
NS = 1                          # one segment is enough for the census
GAIN_MODES = e.GAIN_MODES       # ('FH','FM','FL','AHL_H','AML_M','AHL_L','AML_L')

# Canonical control word for each gain range (independent of pscalib) -- the
# value the decode must map to that range.  Verified below against pscalib and,
# by construction, against the psana bit rule reproduced in _ref_masks().
CANON_CBITS = {"FH": 28, "FM": 12, "FL": 8,
               "AHL_H": 16, "AML_M": 0, "AHL_L": 48, "AML_L": 32}


# --------------------------------------------------------------------------
# minimal stand-ins for psdata's seg_config objects (duck-typed by pscalib):
# seg_configs[seg].config.{trbit, asicPixelConfig}
# --------------------------------------------------------------------------
class _Cfg:
    def __init__(self, trbit, asic_pixel_config):
        self.trbit = trbit
        self.asicPixelConfig = asic_pixel_config


class _Seg:
    def __init__(self, config):
        self.config = config


def _ref_masks(cbits):
    """INDEPENDENT psana bit rule -> 7 boolean gain-range masks (GAIN_MODES
    order).  Reproduces psana ``gain_maps_epix10ka_any_alg`` from the documented
    thresholds; deliberately NOT a call into pscalib's decode, so a divergence
    in pscalib would make this test fail."""
    m60 = cbits & 60
    m28 = cbits & 28
    m12 = cbits & 12
    return [
        (m28 == 28),   # FH    0
        (m28 == 12),   # FM    1
        (m12 == 8),    # FL    2
        (m60 == 16),   # AHL_H 3
        (m60 == 0),    # AML_M 4
        (m60 == 48),   # AHL_L 5
        (m60 == 32),   # AML_L 6
    ]


def _build_synthetic():
    """Build a synthetic single-segment epix10ka that exercises ALL 7 ranges.

    Layout (one panel, four 176x192 ASICs; ASICs 0 & 3 are placed un-flipped by
    the panel reassembly, so their pixels map directly to panel coordinates):

      * trbit = [1,0,0,0]  -> only ASIC 0 (bottom-right quadrant) carries the
        trbit control bit B04 (=16); ASICs 1,2,3 do not.
      * ASIC 0 (bottom-right, b4=1) hosts the three trbit-set ranges by row band:
            FH    (cfg=12, data=0 -> cbits 28)
            AHL_H (cfg=0,  data=0 -> cbits 16)
            AHL_L (cfg=0,  data=1 -> cbits 48)
      * ASIC 3 (bottom-left, b4=0) hosts the four trbit-clear ranges by row band:
            FM    (cfg=12, data=0 -> cbits 12)
            FL    (cfg=8,  data=0 -> cbits 8)
            AML_M (cfg=0,  data=0 -> cbits 0)
            AML_L (cfg=0,  data=1 -> cbits 32)
      * ASICs 1,2 (top half): cfg=0, data=0 -> AML_M (cbits 0); zero is
        flip-invariant so their reassembly orientation is irrelevant here.

    The per-pixel data gain bit is the raw word's bit 14 (B14); config gain bits
    (0b1100) come from asicPixelConfig; the trbit bit comes from ``trbit``.

    Returns (raw, seg_configs, pedestals, pixel_gain, exp_cbits) where exp_cbits
    is the panel control word built purely from the layout above (independent of
    pscalib's cbits machinery).
    """
    rng = np.random.default_rng(20260711)

    apc = np.zeros((4, RH, CH), dtype=np.uint8)
    raw = np.zeros((NS, R, C), dtype=np.uint16)
    # ADC payload in the low 14 bits (< B14 so raw & M14 recovers it exactly).
    raw[0] = rng.integers(1, 16000, size=(R, C), dtype=np.uint16)

    trbit = np.array([1, 0, 0, 0], dtype=np.uint8)

    # bottom-right quadrant (ASIC 0), 3 row bands -> FH / AHL_H / AHL_L
    b = RH // 3
    apc[0][:b, :] = 12        # FH
    apc[0][b:2 * b, :] = 0    # AHL_H
    apc[0][2 * b:, :] = 0     # AHL_L
    raw[0][RH + 2 * b:, CH:] |= e.B14   # data gain bit -> AHL_L band

    # bottom-left quadrant (ASIC 3), 4 row bands -> FM / FL / AML_M / AML_L
    q = RH // 4
    apc[3][:q, :] = 12        # FM
    apc[3][q:2 * q, :] = 8    # FL
    apc[3][2 * q:3 * q, :] = 0  # AML_M
    apc[3][3 * q:, :] = 0     # AML_L
    raw[0][RH + 3 * q:, :CH] |= e.B14   # data gain bit -> AML_L band

    seg_configs = {0: _Seg(_Cfg(trbit, apc))}

    # Distinct, strictly-positive per-range constants so that selecting the
    # WRONG range plane changes the number (this is what makes the census
    # discriminate on per-range indexing).
    pedestals = rng.uniform(5.0, 500.0, size=(7, NS, R, C)).astype(np.float32)
    pixel_gain = rng.uniform(0.3, 6.0, size=(7, NS, R, C)).astype(np.float32)

    # Independent expected control word from the layout (NOT via pscalib).
    exp_cbits = np.zeros((NS, R, C), dtype=np.int64)   # top half -> 0 (AML_M)
    exp_cbits[0][RH:RH + b, CH:] = CANON_CBITS["FH"]
    exp_cbits[0][RH + b:RH + 2 * b, CH:] = CANON_CBITS["AHL_H"]
    exp_cbits[0][RH + 2 * b:, CH:] = CANON_CBITS["AHL_L"]
    exp_cbits[0][RH:RH + q, :CH] = CANON_CBITS["FM"]
    exp_cbits[0][RH + q:RH + 2 * q, :CH] = CANON_CBITS["FL"]
    exp_cbits[0][RH + 2 * q:RH + 3 * q, :CH] = CANON_CBITS["AML_M"]
    exp_cbits[0][RH + 3 * q:, :CH] = CANON_CBITS["AML_L"]

    return raw, seg_configs, pedestals, pixel_gain, exp_cbits


def _protected_inv(gain):
    """1/gain with 0 -> 0 (matches psana ``divide_protected`` / pscalib
    ``gain_factor_from_gain``); reproduced here to keep the expected image
    independent of the library helper."""
    gain = np.asarray(gain, dtype=np.float32)
    return np.divide(np.ones_like(gain), gain,
                     out=np.zeros_like(gain, dtype=np.float32),
                     where=gain != 0).astype(np.float32)


def _hand_expected(raw, pedestals, pixel_gain, masks):
    """The documented per-range calib, computed from INDEPENDENT masks::

        (raw & M14 - pedestals[range]) * (1/pixel_gain[range])

    driven by ``masks`` (psana bit rule), mirroring the exact float32 op order
    of pscalib so the comparison can be byte-exact (np.array_equal)."""
    gfac = _protected_inv(pixel_gain)
    factor = np.select(masks, [gfac[k] for k in range(7)],
                       default=1).astype(np.float32)
    pedest = np.select(masks, [pedestals[k] for k in range(7)],
                       default=0).astype(np.float32)
    arrf = np.array(raw & e.M14, dtype=np.float32)
    arrf = arrf - pedest
    return (arrf * factor).astype(np.float32)


# ==========================================================================
# tests (pytest-collectable AND driven by main())
# ==========================================================================
def test_all_seven_ranges_exercised():
    """The synthetic layout populates ALL 7 gain ranges and they partition the
    panel -- the census a run collapsed to FM (COR-01b) would fail."""
    _, _, _, _, exp_cbits = _build_synthetic()
    masks = _ref_masks(exp_cbits)
    counts = {GAIN_MODES[k]: int(masks[k].sum()) for k in range(7)}
    print("[census] pixels per gain range:", counts)
    for k in range(7):
        assert counts[GAIN_MODES[k]] > 0, \
            f"gain range {GAIN_MODES[k]} was never populated -- census incomplete"
    # exactly one range per pixel (a clean partition, no pixel in >1 or 0 ranges)
    covered = sum(m.astype(np.int64) for m in masks)
    assert covered.min() == 1 and covered.max() == 1, \
        "gain-range masks do not partition the panel (overlap or gap)"
    assert covered.sum() == NS * R * C
    print(f"[ok] all 7 gain ranges exercised; {NS * R * C} pixels partitioned "
          "one-per-range")


def test_pscalib_cbits_matches_layout():
    """pscalib's control-word pipeline (config reassembly + per-event data bit)
    reproduces the layout-derived control word exactly -- confirms the synthetic
    inputs select the ranges we intend before we judge the decode."""
    raw, seg_configs, _, _, exp_cbits = _build_synthetic()
    cbits_cfg = e.cbits_config_detector(seg_configs)
    cbits = e.cbits_config_and_data(raw, cbits_cfg)
    assert cbits.shape == (NS, R, C), cbits.shape
    assert np.array_equal(cbits, exp_cbits), \
        "pscalib control word != layout-derived control word"
    # each canonical range value is actually present in the pipeline's cbits
    present = set(np.unique(cbits).tolist())
    for name, val in CANON_CBITS.items():
        assert val in present, f"{name} control word {val} missing from cbits"
    print("[ok] pscalib cbits pipeline == layout; all 7 canonical words present")


def test_decode_masks_match_psana_reference():
    """pscalib's ``gain_maps_epix10ka_any`` equals the INDEPENDENT psana bit rule
    for every range -- the heart of COR-01b, checked per range, not just FM."""
    raw, seg_configs, _, _, exp_cbits = _build_synthetic()
    cbits = e.cbits_config_and_data(raw, e.cbits_config_detector(seg_configs))
    ps_gmaps = e.gain_maps_epix10ka_any(cbits)
    ref = _ref_masks(exp_cbits)
    for k in range(7):
        assert np.array_equal(np.asarray(ps_gmaps[k]), ref[k]), \
            f"pscalib decode mask for {GAIN_MODES[k]} != psana reference"
        assert ref[k].any(), f"{GAIN_MODES[k]} not exercised"
    print("[ok] pscalib decode masks match the psana bit rule for all 7 ranges")


def test_calib_matches_hand_derived_all_ranges():
    """End-to-end: ``calib_epix10ka`` output is byte-exact to the hand-derived
    per-range image, AND each range is verified BY HAND at a representative
    pixel with the scalar formula (raw-ped)/gain."""
    raw, seg_configs, pedestals, pixel_gain, exp_cbits = _build_synthetic()
    masks = _ref_masks(exp_cbits)

    got = e.calib_epix10ka(raw, pedestals, pixel_gain, seg_configs)
    assert got.shape == (NS, R, C) and got.dtype == np.float32, \
        (got.shape, got.dtype)

    expected = _hand_expected(raw, pedestals, pixel_gain, masks)
    d = np.abs(got - expected)
    assert np.array_equal(got, expected), \
        f"calib not byte-exact vs hand-derived per-range image: max|diff|={d.max()}"

    # explicit per-range BY-HAND scalar check at one representative pixel each,
    # proving the range index -> constant plane wiring for every range.
    gidx = np.select(masks, list(range(7)), default=-1)
    for k in range(7):
        ys, xs = np.where(gidx[0] == k)
        assert ys.size > 0, f"{GAIN_MODES[k]} has no pixel"
        i, j = int(ys[0]), int(xs[0])
        adc = np.float32(raw[0, i, j] & e.M14)
        ped = np.float32(pedestals[k, 0, i, j])
        gain = np.float32(pixel_gain[k, 0, i, j])
        hand = np.float32(adc - ped) * (np.float32(1.0) / gain)
        assert np.isclose(got[0, i, j], hand, rtol=0, atol=1e-3), \
            (f"{GAIN_MODES[k]} pixel ({i},{j}): calib={got[0, i, j]} != "
             f"(adc {adc} - ped {ped}) / gain {gain} = {hand}")
    print("[ok] calib byte-exact vs hand-derived image; per-range scalar formula "
          "verified for all 7 ranges (max|diff|=%g)" % d.max())

    # per-range output means are distinct -> ranges are genuinely differentiated
    means = [float(got[masks[k]].mean()) for k in range(7)]
    assert len(set(round(m, 3) for m in means)) == 7, \
        f"per-range means not distinct -- ranges may be collapsing: {means}"


def test_range_selection_is_load_bearing():
    """Negative control: if any range's pedestal plane were mis-assigned (here we
    swap the AHL_H and AML_M planes), the output WOULD differ -- so the byte-exact
    match above is discriminating, not vacuous.  (No source is changed; this only
    perturbs the EXPECTED to show the test's sensitivity.)"""
    raw, seg_configs, pedestals, pixel_gain, exp_cbits = _build_synthetic()
    masks = _ref_masks(exp_cbits)
    got = e.calib_epix10ka(raw, pedestals, pixel_gain, seg_configs)

    swapped = pedestals.copy()
    ahl_h, aml_m = GAIN_MODES.index("AHL_H"), GAIN_MODES.index("AML_M")
    swapped[[ahl_h, aml_m]] = swapped[[aml_m, ahl_h]]
    wrong = _hand_expected(raw, swapped, pixel_gain, masks)
    assert not np.array_equal(got, wrong), \
        "swapping two range planes did not change the result -- test is vacuous"
    # and the difference is confined to exactly the two swapped ranges' pixels
    diff = ~np.isclose(got, wrong, rtol=0, atol=0)
    affected = masks[ahl_h] | masks[aml_m]
    assert np.array_equal(diff, affected), \
        "mis-assigned planes changed pixels outside the two swapped ranges"
    print("[ok] negative control: per-range plane selection is load-bearing "
          "(swap changes exactly the AHL_H/AML_M pixels)")


def test_registry_dispatch_matches_leaf():
    """The public registry entry (``pscalib.calib`` for the epix10ka class name)
    produces the same all-7-range result as the leaf ``calib_epix10ka``."""
    import pscalib
    raw, seg_configs, pedestals, pixel_gain, _ = _build_synthetic()
    leaf = e.calib_epix10ka(raw, pedestals, pixel_gain, seg_configs)
    via_reg = pscalib.calib(
        "epix10ka_raw_2_0_1", raw,
        {"pedestals": pedestals, "pixel_gain": pixel_gain},
        config=seg_configs)
    assert np.array_equal(via_reg, leaf), "registry dispatch != leaf calib"
    print("[ok] registry dispatch reproduces the leaf 7-range calib")


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("COR-01b: epix10ka full 7-gain-range census (FH FM FL AHL_H AML_M "
          "AHL_L AML_L)")
    print("parent @ 6d7870a754b68c991dc1df484d2543ac4cf856ca  (case A: coverage "
          "gap, no source change)")
    print("=" * 72)
    test_all_seven_ranges_exercised()
    test_pscalib_cbits_matches_layout()
    test_decode_masks_match_psana_reference()
    test_calib_matches_hand_derived_all_ranges()
    test_range_selection_is_load_bearing()
    test_registry_dispatch_matches_leaf()
    print("\nALL COR-01b GAIN-CENSUS CHECKS PASSED "
          "(all 7 gain-range decode branches exercised)")


if __name__ == "__main__":
    main()
