#!/usr/bin/env python3
"""US-008 acceptance test: synthetic mixed-trbit epix10ka byte-exact gate.

Covers the per-quadrant trbit-OR ELSE branch of the epix10ka gain-range decode
(``cbits_config_epix10ka``), which the US-004 byte-exact proof could NOT reach.

WHY SYNTHETIC CONFIGS (from the 2026-06-16 trbit hunt)
-----------------------------------------------------
The US-004 dataset (exp=ued1010667, run=177, det='epixquad') has every segment
at ``trbit=[0,0,0,0]`` -- which hits the ``not any(trbits): return cbits``
EARLY-RETURN in psana ``UtilsEpix10ka.cbits_config_epix10ka`` (and pscalib's
faithful twin).  A uniform ``trbit=[1,1,1,1]`` hits the ``all(trbits)``
whole-array shortcut.  The per-quadrant trbit-OR ELSE branch -- which OR-s the
trbit B04 control bit into ONE 176x192 ASIC quadrant at a time, with the
quadrant<->ASIC permutation

    trbit[2] -> top-left      cbits[:176, :192]
    trbit[3] -> bottom-left   cbits[176:, :192]
    trbit[0] -> bottom-right  cbits[176:, 192:]
    trbit[1] -> top-right     cbits[:176, 192:]

-- is therefore ONLY exercised by a MIXED trbit pattern.

No epix10ka-class dataset with a non-zero trbit exists in accessible SDF data,
and NO dataset of any class has a MIXED trbit (search trail: public01/xtc,
/sdf/data/lcls/ds/ued/* ~40 exps, mfx/xpp/cxi/rix/tmo; UED runs the quad in
fixed gain).  The only non-zero trbit anywhere is rixx45619 r121/122 det=epixhr,
but that is an epixhr2x2 (different code path ``cbits_config_epixhr2x2``, shape
288x384, different quadrant map) with uniform ``[1,1,1,1]`` and broken calib.
So "find a dataset" is a dead end -- this gate uses MUTATED configs instead.

THE GATE
--------
We load the REAL ued1010667/r177 epixquad ``det.raw._seg_configs()`` as a
template (real ``asicPixelConfig``), the REAL snapshot constants (pedestals /
pixel_gain / mask / pixel_status), and one REAL raw event.  Then for a battery
of trbit patterns we programmatically mutate ``cfg.config.trbit`` and assert,
byte-for-byte (``np.array_equal``), that pscalib's per-pixel cbits map AND the
resulting calib over the fixed raw input equal psana's
``cbits_config_epix10ka`` / ``calib_epix10ka_any`` math driven by the SAME
mutated config objects, in the psconda.sh env.

The psana ORACLE is built from psana's OWN functions, so the comparison is a
true psana-vs-pscalib byte check (not pscalib compared to itself):

    cbits_seg   = U.cbits_config_epix10ka(mutated_cob)          # per segment
    cbits_det   = np.stack(cbits_seg over segments)             # _cbits_config_detector
    cbits_total = U.cbits_config_and_data_detector_alg(raw, cbits_det, B14, 9)
    gmaps       = U.gain_maps_epix10ka_any_alg(cbits_total)
    factor      = U.event_constants_for_gmaps(gmaps, gfac, default=1)
    pedest      = U.event_constants_for_gmaps(gmaps, peds,  default=0)
    calib       = (raw & M14 - pedest) * factor * mask          # calib_epix10ka_any body

The mixed patterns are checked to OBSERVABLY exercise the ELSE branch: the four
176x192 quadrants of the cbits map differ according to the trbit permutation
(not a uniform fill).

EpixHR2x2 / EpixHR1x4 use DIFFERENT quadrant maps and are explicitly OUT OF
SCOPE for this gate (this gate is epix10ka-352x384-only).

Run on sdfiana025 via ``run_tests.sh tests/test_epix10ka_trbit_us008.py`` (it
needs the PRODUCTION psana env -- psconda.sh -- for the oracle and to read the
template config; the offline import-purity check runs without psana).
"""

import copy
import os
import subprocess
import sys
import tempfile

import numpy as np

# --- locate the pscalib package (parent of this tests dir) ------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- the TEMPLATE for the synthetic configs.
EXP = "ued1010667"
RUN = 177
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "epixquad"
SEGS = [0, 1, 2, 3]

RAW_SHAPE = (4, 352, 384)
CALIB_SHAPE = (4, 352, 384)
CONS_SHAPE = (7, 4, 352, 384)          # leading 7 = epix10ka gain ranges
ROWSH, COLSH = 176, 192                 # ASIC half-panel (the quadrant size)

# The forbidden set (shared with pscalib._purity).
FORBIDDEN = ("psana", "mpi4py", "h5py", "dgram", "pymongo")

# The trbit battery: 5 MIXED patterns that exercise the per-quadrant ELSE
# branch, plus the two CONTROLS (all-zero early-return, all-one whole-array).
MIXED_PATTERNS = [
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (1, 1, 0, 0),
    (0, 0, 1, 1),
    (1, 0, 0, 1),
]
CONTROL_PATTERNS = [
    (0, 0, 0, 0),      # not any(trbits) -> early return
    (1, 1, 1, 1),      # all(trbits) -> whole-array shortcut
]
ALL_PATTERNS = MIXED_PATTERNS + CONTROL_PATTERNS

# Quadrant<->ASIC(trbit) permutation, in panel coordinates.  Each entry maps a
# trbit index to the (row-slice, col-slice) of the 176x192 quadrant that the
# psana ELSE branch OR-s B04 into when that trbit is set.
QUADRANT_FOR_TRBIT = {
    2: (slice(None, ROWSH), slice(None, COLSH)),   # top-left
    3: (slice(ROWSH, None), slice(None, COLSH)),   # bottom-left
    0: (slice(ROWSH, None), slice(COLSH, None)),   # bottom-right
    1: (slice(None, ROWSH), slice(COLSH, None)),   # top-right
}
B04 = 0o20  # 16 -- the trbit control bit


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


def _have_psdata():
    try:
        import psdata  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# OFFLINE (no psana): purely-synthetic mixed-trbit cbits + calib + the ELSE
# branch geometry assertion.  This is the core US-008 logic and runs without
# psana -- it builds its own raw/constants and drives pscalib alone, but the
# EXPECTED cbits is computed by an independent in-test re-derivation of the
# psana per-quadrant rule (so a regression in pscalib's quadrant map fails it).
# The psana-vs-pscalib byte gate (below) is the authoritative oracle check.
# --------------------------------------------------------------------------
def _expected_cbits_seg(trbit, apc):
    """Independent re-derivation of psana's per-quadrant trbit-OR rule for one
    segment, used to (a) assert pscalib matches and (b) prove the ELSE branch
    is observably exercised.  Mirrors psana cbits_config_epix10ka exactly."""
    pca = np.asarray(apc)
    cbits = np.vstack((
        np.hstack((np.flipud(np.fliplr(pca[2])), np.flipud(np.fliplr(pca[1])))),
        np.hstack((pca[3], pca[0])),
    ))
    cbits = np.bitwise_and(cbits, 12)
    trbits = tuple(int(t) for t in trbit)
    if all(trbits):
        return np.bitwise_or(cbits, B04)
    if not any(trbits):
        return cbits
    for ti in (2, 3, 0, 1):
        if trbits[ti]:
            rs, csl = QUADRANT_FOR_TRBIT[ti]
            cbits[rs, csl] = np.bitwise_or(cbits[rs, csl], B04)
    return cbits


def _assert_else_branch_exercised(cbits_seg, trbit):
    """For a MIXED pattern, assert each 176x192 quadrant's B04 (trbit) bit is
    present iff that quadrant's trbit is set -- i.e. the per-quadrant ELSE
    branch produced a NON-uniform fill matching the permutation."""
    trbits = tuple(int(t) for t in trbit)
    for ti, (rs, csl) in QUADRANT_FOR_TRBIT.items():
        quad = cbits_seg[rs, csl]
        has_trbit = bool((quad & B04).any())
        # asicPixelConfig & 12 can never set B04 (16), so B04 presence is a
        # clean signal of the trbit-OR for this quadrant.
        assert has_trbit == bool(trbits[ti]), (
            f"trbit{ti}={trbits[ti]} but quadrant B04-present={has_trbit} "
            f"-- per-quadrant ELSE branch mis-mapped")


def test_pscalib_cbits_matches_quadrant_rule_offline():
    """OFFLINE: pscalib's cbits_config_epix10ka equals an independent
    re-derivation of psana's per-quadrant rule for EVERY pattern, and the
    mixed patterns observably exercise the ELSE branch (quadrants differ)."""
    from pscalib.apply.epix10ka import cbits_config_epix10ka

    rng = np.random.default_rng(8)
    # asicPixelConfig values that exercise the &12 gain config bits.
    apc = rng.integers(0, 16, size=(4, ROWSH, COLSH), dtype=np.uint8)

    for trbit in ALL_PATTERNS:
        got = cbits_config_epix10ka(np.asarray(trbit, np.uint8), apc)
        exp = _expected_cbits_seg(trbit, apc)
        assert got.shape == (352, 384), got.shape
        assert np.array_equal(got, exp), (
            f"pscalib cbits != quadrant rule for trbit={trbit}")
        if trbit in MIXED_PATTERNS:
            _assert_else_branch_exercised(got, trbit)
            # extra: the four quadrants are NOT a uniform fill -- at least one
            # has B04 and at least one does not (true for every MIXED pattern).
            present = [bool((got[rs, csl] & B04).any())
                       for rs, csl in QUADRANT_FOR_TRBIT.values()]
            assert any(present) and not all(present), (
                f"mixed trbit={trbit} did not produce a non-uniform cbits map")
    print(f"[ok] OFFLINE: pscalib cbits matches per-quadrant rule for "
          f"{len(ALL_PATTERNS)} patterns; ELSE branch exercised for "
          f"{len(MIXED_PATTERNS)} mixed patterns")


def test_offline_apply_purity_subprocess():
    """In a FRESH interpreter: build mutated mixed-trbit configs and run the
    epix10ka apply through the registry.  None of the forbidden modules may
    appear; numpy must.  Mirrors US-004's purity gate for the mixed path."""
    script = (
        "import sys\n"
        "import numpy as np\n"
        "import pscalib\n"
        "class _Ns:\n"
        "    def __init__(self, t, a):\n"
        "        self.trbit = t; self.asicPixelConfig = a\n"
        "class _Seg:\n"
        "    def __init__(self, c):\n"
        "        self.config = c\n"
        "ped = np.zeros((7, 4, 352, 384), np.float32)\n"
        "gain = np.ones((7, 4, 352, 384), np.float32)\n"
        "raw = np.zeros((4, 352, 384), np.uint16)\n"
        "trbit = np.array([1, 0, 1, 0], np.uint8)\n"
        "apc = np.zeros((4, 176, 192), np.uint8)\n"
        "cfg = {i: _Seg(_Ns(trbit, apc)) for i in range(4)}\n"
        "cons = {'pedestals': ped, 'pixel_gain': gain}\n"
        "out = pscalib.calib('epix10ka_raw_2_0_1', raw, cons, config=cfg)\n"
        "assert out.shape == (4, 352, 384) and out.dtype == np.float32\n"
        "pscalib.assert_no_framework_imports()\n"
        "bad = [m for m in %r if m in sys.modules]\n"
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
    print("[ok] offline apply import purity (subprocess, mixed-trbit config)")


# --------------------------------------------------------------------------
# psana ORACLE: byte-exact pscalib-vs-psana for every mutated pattern.
# --------------------------------------------------------------------------
class _Cob:
    """A minimal stand-in for psana's per-segment config object -- exposes
    exactly the two fields ``cbits_config_epix10ka(cob)`` reads."""
    def __init__(self, trbit, asic_pixel_config):
        self.trbit = np.asarray(trbit)
        self.asicPixelConfig = np.asarray(asic_pixel_config)


class _Seg:
    """A stand-in for a psana ``_seg_configs()[i]`` entry: ``.config`` is the
    cob.  pscalib's ``cbits_config_detector`` duck-types on exactly this."""
    def __init__(self, cob):
        self.config = cob


def _psana_oracle_cbits_and_calib(U, mut_cobs, raw, peds, gfac, mask):
    """Build psana's OWN cbits map and final calib for the mutated configs,
    using psana's exact functions (the body of ``calib_epix10ka_any``)."""
    # per-segment cbits, stacked in segment order (psana _cbits_config_detector)
    cbits_det = np.stack(tuple(U.cbits_config_epix10ka(c) for c in mut_cobs))
    # + per-event data gain bit (B14 -> B05), epix10ka params
    cbits_total = U.cbits_config_and_data_detector_alg(
        raw, cbits_det, U.B14, 9)
    gmaps = U.gain_maps_epix10ka_any_alg(cbits_total)
    factor = U.event_constants_for_gmaps(gmaps, gfac, default=1)
    pedest = U.event_constants_for_gmaps(gmaps, peds, default=0)
    raw14 = np.bitwise_and(raw, U.M14)
    arrf = np.array(raw14, dtype=np.float32)
    if pedest is not None:
        arrf -= pedest
    calib = arrf * factor if mask is None else arrf * factor * mask
    return cbits_det, calib.astype(np.float32)


def test_mixed_trbit_byte_exact(out_dir):
    """For EACH trbit pattern, pscalib's cbits map and calib (over the real raw)
    are byte-identical to psana's, driven by the SAME mutated config objects."""
    import psana.detector.UtilsEpix10ka as U
    import pscalib
    from pscalib.apply import epix10ka as pe
    import pscalib.providers.snapshot as ps_snap

    # --- real constants (one-time snapshot) -----------------------------
    snap_dir = ps_snap.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                      out_dir=out_dir)
    snap = ps_snap.load_snapshot(snap_dir)
    peds = np.asarray(snap.pedestals, dtype=np.float32)
    gain = np.asarray(snap.pixel_gain, dtype=np.float32)
    mask = None if snap.mask is None else np.asarray(snap.mask)
    assert peds.shape == CONS_SHAPE and gain.shape == CONS_SHAPE, \
        (peds.shape, gain.shape)
    gfac = pe.gain_factor_from_gain(gain)   # 1/gain (protected) -- as psana store
    print(f"[snap] pedestals {peds.shape} pixel_gain {gain.shape} "
          f"mask {None if mask is None else mask.shape}")

    # --- real template configs (the asicPixelConfig we MUTATE trbit on) --
    # read seg_configs offline via psdata (the US-003 plumbing); fall back to
    # psana det.raw._seg_configs() if psdata is unavailable.
    template_apc = {}     # {seg: asicPixelConfig (4,176,192)}
    raw = None
    ts64 = None
    if _have_psdata():
        import psdata
        run = psdata.open(exp=EXP, run=RUN, dir=DIR)
        seg_cfg = run.seg_configs(DET)
        assert sorted(seg_cfg) == SEGS, sorted(seg_cfg)
        for s in SEGS:
            template_apc[s] = np.asarray(seg_cfg[s].config.asicPixelConfig)
    # get one real raw event + its asicPixelConfig from psana (also validates
    # the psdata template matches psana's).
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    evt = next(myrun.events())
    ts64 = evt.timestamp() if callable(getattr(evt, "timestamp", None)) \
        else evt.timestamp
    raw = np.asarray(det.raw.raw(evt))
    assert raw.shape == RAW_SHAPE and raw.dtype == np.uint16, raw.shape
    psana_seg = det.raw._seg_configs()
    for s in SEGS:
        psana_apc = np.asarray(psana_seg[s].config.asicPixelConfig)
        if s in template_apc:
            assert np.array_equal(template_apc[s], psana_apc), \
                f"psdata seg_config apc != psana for seg {s}"
        else:
            template_apc[s] = psana_apc
    # sanity: the real run is all-zero trbit (the untested-branch motivation).
    real_trbit = [tuple(int(t) for t in np.asarray(psana_seg[s].config.trbit))
                  for s in SEGS]
    print(f"[template] raw {raw.shape} ts={ts64}; real trbit per seg="
          f"{real_trbit} (all-zero -> ELSE branch never reached by US-004)")

    n_patterns = 0
    for trbit in ALL_PATTERNS:
        # build mutated configs for BOTH sides from the SAME template apc
        mut_cobs = [_Cob(np.asarray(trbit, dtype=np.asarray(
            psana_seg[s].config.trbit).dtype), template_apc[s])
            for s in SEGS]
        pscalib_cfg = {s: _Seg(mut_cobs[i]) for i, s in enumerate(SEGS)}

        # --- psana oracle ----------------------------------------------
        ora_cbits, ora_calib = _psana_oracle_cbits_and_calib(
            U, mut_cobs, raw, peds, gfac, mask)

        # --- pscalib ---------------------------------------------------
        my_cbits = pe.cbits_config_detector(pscalib_cfg, segment_ids=SEGS)
        my_calib = pscalib.calib("epix10ka_raw_2_0_1", raw,
                                 {"pedestals": peds, "pixel_gain": gain},
                                 config=pscalib_cfg, mask=mask) \
            if _calib_accepts_mask() else \
            pe.calib_epix10ka(raw, peds, gain, pscalib_cfg, mask=mask,
                              segment_ids=SEGS)

        assert my_cbits.shape == (4, 352, 384), my_cbits.shape
        assert np.array_equal(my_cbits, ora_cbits), (
            f"cbits not byte-exact for trbit={trbit}: "
            f"max|diff|={np.abs(my_cbits.astype(int) - ora_cbits.astype(int)).max()}")
        assert my_calib.shape == CALIB_SHAPE and my_calib.dtype == np.float32
        d = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(ora_calib))
        assert np.array_equal(my_calib, ora_calib), (
            f"calib not byte-exact for trbit={trbit}: max|diff|={d.max()}")

        # the mixed patterns must observably exercise the per-quadrant ELSE
        # branch -- assert on the PER-SEGMENT cbits the quadrants differ.
        if trbit in MIXED_PATTERNS:
            for s_i in range(4):
                _assert_else_branch_exercised(my_cbits[s_i], trbit)
        print(f"[byte-exact] trbit={trbit}: cbits + calib array_equal=True "
              f"(max|calib diff|={d.max()})")
        n_patterns += 1

    assert n_patterns == len(ALL_PATTERNS), n_patterns
    print(f"[ok] (a) mixed-trbit byte-exact gate: {n_patterns} patterns "
          f"({len(MIXED_PATTERNS)} mixed + {len(CONTROL_PATTERNS)} controls), "
          f"cbits + calib max|diff| == 0 vs psana")
    return snap_dir


def _calib_accepts_mask():
    """The registry ``pscalib.calib`` accepts a ``mask=`` kwarg (US-005). Probe
    once so this test works whether or not that kwarg exists."""
    import inspect
    import pscalib.registry as reg
    try:
        params = inspect.signature(reg.calib).parameters
    except (TypeError, ValueError):
        return False
    return "mask" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-008 acceptance: synthetic mixed-trbit epix10ka byte-exact gate")
    print("=" * 72)

    # offline checks always run (no psana needed)
    test_pscalib_cbits_matches_quadrant_rule_offline()
    test_offline_apply_purity_subprocess()

    if not _have_psana():
        print("\n[skip] psana not importable -- byte-exact oracle gate skipped. "
              "Source psconda.sh on sdfiana025.")
        print("\nUS-008 offline checks PASSED (psana oracle gate skipped)")
        return

    tmp = tempfile.mkdtemp(prefix="pscalib_us008_")
    try:
        test_mixed_trbit_byte_exact(out_dir=tmp)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL US-008 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
