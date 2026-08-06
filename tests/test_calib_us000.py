#!/usr/bin/env python3
"""US-000 acceptance test: scaffold pscalib + migrate psdata's jungfrau calib
+ image-assembly into it (non-regressing).

Verifies the US-000 acceptance criteria for the reference Jungfrau dataset
(exp=mfx100848724, run=51, dir=/sdf/data/lcls/ds/prj/public01/xtc, det=jungfrau):

  (b) NON-REGRESSION -- render byte-exact vs psana.  The lifted render produces
      calib (32,512,1024) f32 AND assembled image (4216,4432) f32 with
      max|diff| == 0 vs ``det.raw.calib(evt)`` / ``det.raw.image(evt)`` for one
      event whose 64-bit timestamp comes from psana itself (``evt.timestamp()``).

  (c) IMPORT PURITY -- the extended forbidden set.  pscalib's
      ``assert_no_framework_imports()`` forbids
      ('psana','mpi4py','h5py','dgram','pymongo') -- it EXTENDS psdata's set
      (which omits dgram + pymongo).  After importing pscalib, reloading a
      snapshot, and running the jungfrau apply in a FRESH interpreter, none of
      those five appear in sys.modules.

This test needs the PRODUCTION psana env (the psconda.sh install) to GENERATE
the snapshots + ground truth -- run it on sdfiana025 via
``run_tests.sh tests/test_calib_us000.py``.  The byte-exact + non-regression
checks skip cleanly (with a message) if psana is not importable; the offline
import-purity checks still run without the prod env.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# --- machine-readable skip protocol (HYG-05); see tests/_skips.py -----------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from _skips import skip  # noqa: E402

# --- locate the pscalib package (parent of this tests dir) ------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- lives in the TEST, never in the library.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "jungfrau"

# Per-acceptance expected shapes/dtypes of the named gain-calibration constants.
EXPECT = {
    "pedestals":    ((3, 32, 512, 1024), np.float32),
    "pixel_gain":   ((3, 32, 512, 1024), np.float32),
    "pixel_offset": ((3, 32, 512, 1024), np.float32),
    "mask":         ((32, 512, 1024),    np.uint8),
}

# The forbidden set pscalib EXTENDS to (psdata's was the first three only).
FORBIDDEN = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1_dir(snap_dir):
    """sha1 every regular file in a snapshot dir, keyed by relative name."""
    out = {}
    for name in sorted(os.listdir(snap_dir)):
        p = os.path.join(snap_dir, name)
        if os.path.isfile(p):
            out[name] = _sha1(p)
    return out


# --------------------------------------------------------------------------
# (c) import purity of the OFFLINE path -- pure numpy, extended forbidden set
# --------------------------------------------------------------------------
def test_offline_import_purity_in_proc():
    """Importing pscalib (apply + snapshot reload + render engine) must not
    pull in any framework.  The psana touches (snapshot capture, geometry
    derivation) import psana lazily, on call only."""
    import pscalib
    _ = (pscalib.Imager, pscalib.calib_jungfrau, pscalib.assemble_image,
         pscalib.load_snapshot, pscalib.CalibSnapshot)
    # pscalib's forbidden set is the extended 5-tuple
    assert pscalib.FORBIDDEN_MODULES == FORBIDDEN, pscalib.FORBIDDEN_MODULES
    pscalib.assert_no_framework_imports()
    for m in FORBIDDEN:
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"


def test_offline_import_purity_subprocess(snapshot_dir=None):
    """In a FRESH interpreter: import pscalib, reload a snapshot, and run the
    jungfrau apply (raw->calib->image).  None of the five forbidden modules may
    appear; numpy must.  This is the US-000 (c) gate."""
    apply_stmt = ""
    if snapshot_dir:
        apply_stmt = (
            "import numpy as np; "
            "from pscalib import load_snapshot, Imager; "
            f"snap=load_snapshot({snapshot_dir!r}); "
            "im=Imager(snap, derive_geometry_if_missing=False); "
            # synthetic raw: shape from the cached mask, dtype uint16
            "nseg=snap.mask.shape[0]; "
            "raw=np.zeros((nseg,512,1024), dtype=np.uint16); "
            "calib=im.calib(raw); img=im.image(calib); "
            "assert calib.shape==(nseg,512,1024) and calib.dtype==np.float32; "
            "assert img.ndim==2 and img.dtype==np.float32; "
        )
    code = (
        "import sys, pscalib; "
        + apply_stmt +
        "pscalib.assert_no_framework_imports(); "
        f"bad=[m for m in {FORBIDDEN!r} if m in sys.modules]; "
        "assert not bad, bad; "
        "assert 'numpy' in sys.modules, 'numpy should be imported'; "
        "print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=_PKG_PARENT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


# --------------------------------------------------------------------------
# (b) non-regression: render byte-exact vs psana for an evt.timestamp() event
# --------------------------------------------------------------------------
def test_render_byte_exact(out_dir):
    """Snapshot the reference run + cache index maps (psana), then render
    raw->calib->image fully offline and assert byte-identical to psana for ONE
    event whose 64-bit timestamp comes from psana itself."""
    import pscalib
    import pscalib.providers.snapshot as ps_snap
    import pscalib.geometry as pgeo

    # --- regenerate psana ground truth ourselves ------------------------
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    evt = next(myrun.events())
    # The 64-bit timestamp from psana itself.  In psana2 ``Event.timestamp`` is
    # an int attribute (older API exposed it as a method); accept either.
    ts64 = evt.timestamp() if callable(getattr(evt, "timestamp", None)) \
        else evt.timestamp
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_calib = np.asarray(det.raw.calib(evt))
    gt_image = np.asarray(det.raw.image(evt))
    assert gt_raw.shape == (32, 512, 1024) and gt_raw.dtype == np.uint16
    assert gt_calib.shape == (32, 512, 1024) and gt_calib.dtype == np.float32
    print(f"[gt] ts={ts64} raw {gt_raw.shape} calib {gt_calib.shape} "
          f"image {gt_image.shape}")

    # --- one-time snapshot of constants + geometry index maps -----------
    snap_dir = ps_snap.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                      out_dir=out_dir)
    ix_path, iy_path = pgeo.cache_pixel_indexes_for_snapshot(snap_dir)
    print(f"[prep] cached index maps:\n  {ix_path}\n  {iy_path}")

    # --- everything below is the pure-numpy offline render --------------
    snap = ps_snap.load_snapshot(snap_dir)
    imager = pscalib.Imager(snap, derive_geometry_if_missing=False)
    print(f"[render] {imager!r}")

    my_calib = imager.calib(gt_raw)
    assert my_calib.shape == (32, 512, 1024), my_calib.shape
    assert my_calib.dtype == np.float32, my_calib.dtype
    dcal = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(gt_calib))
    assert np.array_equal(my_calib, gt_calib), (
        f"calib not byte-exact: max|diff|={dcal.max()}")
    print(f"[byte-exact] calib {my_calib.shape} {my_calib.dtype} "
          f"max|diff|={dcal.max()} array_equal=True")

    my_image = imager.image(my_calib)
    assert my_image.ndim == 2 and my_image.dtype == np.float32
    di = np.abs(np.nan_to_num(my_image) - np.nan_to_num(gt_image))
    assert np.array_equal(my_image, gt_image), (
        f"image not byte-exact: max|diff|={di.max()}")
    print(f"[byte-exact] image {my_image.shape} {my_image.dtype} "
          f"max|diff|={di.max()} array_equal=True")

    # render() convenience == the two steps
    c2, i2 = imager.render(gt_raw)
    assert np.array_equal(c2, my_calib) and np.array_equal(i2, my_image)

    # index maps derived from geometry text == psana _pixel_coord_indexes
    pix = det.raw._pixel_coord_indexes()
    gt_ix, gt_iy = np.asarray(pix[0]), np.asarray(pix[1])
    assert np.array_equal(imager.ix, gt_ix), "ix != psana _pixel_coord_indexes"
    assert np.array_equal(imager.iy, gt_iy), "iy != psana _pixel_coord_indexes"
    print("[geo] cached ix/iy == det.raw._pixel_coord_indexes() (byte-exact)")

    return snap_dir


# --------------------------------------------------------------------------
# reload byte-exact vs psana _calibconst (the calib half, like US-006)
# --------------------------------------------------------------------------
def test_snapshot_reload_byte_exact(out_dir):
    """Snapshot the reference Jungfrau run, reload offline, and assert the
    reloaded arrays are byte-identical to psana's _calibconst / _mask, with the
    expected shapes, a correct pin, and retained validity metadata."""
    import pscalib.providers.snapshot as ps_snap

    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    gt_cc = det.raw._calibconst                  # {ctype:(ndarray|str, meta)}
    gt_mask = np.asarray(det.raw._mask(status=True))
    gt_uniqueid = det.raw._uniqueid

    assert gt_cc is not None, "psana _calibconst is None (DB unreachable?)"
    for ctype in ("pedestals", "pixel_gain", "pixel_offset"):
        assert ctype in gt_cc, f"psana did not return {ctype!r}"

    snap_dir = ps_snap.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                      out_dir=out_dir)
    assert os.path.basename(snap_dir) == f"{DET}_r{RUN:04d}", snap_dir

    snap = ps_snap.load_snapshot(snap_dir)
    print(f"[reload] {snap!r}")

    assert snap.run == RUN and snap.detname == DET
    assert snap.detector_uniqueid == gt_uniqueid, "pin uniqueid != psana _uniqueid"
    assert snap.exp == EXP

    rebuilt = snap.calibconst()
    for ctype, (gt_arr, _gt_meta) in gt_cc.items():
        if isinstance(gt_arr, np.ndarray):
            got = snap.array(ctype)
            assert got is not None, f"snapshot dropped ndarray ctype {ctype!r}"
            assert got.shape == gt_arr.shape and got.dtype == gt_arr.dtype
            assert np.array_equal(got, gt_arr), f"byte mismatch for {ctype!r}"
            assert np.array_equal(rebuilt[ctype][0], gt_arr)
        elif isinstance(gt_arr, str):
            assert snap.geometry == gt_arr, "geometry text mismatch"
            assert rebuilt[ctype][0] == gt_arr
    assert snap.mask is not None and np.array_equal(snap.mask, gt_mask)

    for ctype, (shape, dtype) in EXPECT.items():
        arr = snap.array(ctype) if ctype != "mask" else snap.mask
        assert arr is not None and arr.shape == shape and arr.dtype == dtype
    assert snap.pedestals.shape[0] == 3
    assert snap.geometry is not None and 1000 < len(snap.geometry) < 20000

    for ctype, (gt_arr, gt_meta) in gt_cc.items():
        v = snap.validity(ctype)
        for k in ("run", "run_end", "version"):
            assert k in v, f"validity missing {k!r} for {ctype!r}"
        if isinstance(gt_meta, dict) and "run" in gt_meta:
            assert int(v["run"]) == int(gt_meta["run"])
    assert snap.is_valid_for_run(RUN)
    print("[reload byte-exact] every ndarray ctype + mask + geometry + "
          "validity match psana (np.array_equal)")
    return snap_dir


# --------------------------------------------------------------------------
# The jungfrau fast kernel's three delicate invariants.
#
# These are OFFLINE (numpy only, synthetic detector, no psana, no SLAC data) and
# they guard three things whose failure mode is SILENT -- each produces a wrong
# number with no exception, and none of them is visible to a gate whose output
# array is contiguous and whose mask never changes:
#
#   1. the sparse gather's C-CONTIGUOUS guard.  ``t.reshape(-1)`` on a
#      non-contiguous view returns a COPY, not a view, so every scattered pixel
#      would be written into a temporary and thrown away -- silently.  ``out=``
#      is a public parameter, so a caller really can hand in a strided view.
#   2. the identity the ``gfm = gfac * mask`` fold is MEMOISED on.  gfm is a
#      function of (pixel_gain, mask); keyed on the gain alone, a caller who
#      changed only the mask would get the previous mask's fold, silently.
#   3. the fold's FAIL-CLOSED guard.  ``(x*g)*m == x*(g*m)`` is not an identity;
#      :func:`pscalib.apply._fastcalib.fold_is_exact` is the theorem's
#      hypothesis, and when it does not hold the kernel must fall back to the
#      unfolded path rather than fold anyway (or raise).
#
# Every assertion below is byte-exactness against
# ``calib_jungfrau_reference`` -- the verbatim c5ce538 expression kept in-tree
# -- under the same three conjuncts the campaign bit gate uses (NaN mask, raw
# uint32 bits, and the SIGN OF EVERY ZERO, which a bare max|diff| cannot see).
# --------------------------------------------------------------------------
_JF_NSEG, _JF_ROWS, _JF_COLS = 2, 512, 1024


def _jf_bits_equal(got, ref, what):
    """Assert byte-exactness of ``got`` vs ``ref``; return a one-line detail.

    Three conjuncts, all required (a bare ``max|diff| == 0`` has two holes: it
    is blind to ``-0.0`` vs ``+0.0``, which the reference really does produce
    for a masked finite-negative pixel, and it is NaN if either side holds a
    NaN, so it fails closed on correct output):

      1. ``isnan(a)`` and ``isnan(b)`` are the same array;
      2. on the non-NaN set, ``a.view(uint32) == b.view(uint32)`` elementwise;
      3. ``signbit(a) == signbit(b)`` everywhere ``a == 0``.
    """
    assert got.dtype == np.float32, (what, got.dtype)
    assert got.shape == ref.shape, (what, got.shape, ref.shape)
    a = np.ascontiguousarray(got).reshape(-1)
    b = np.ascontiguousarray(ref).reshape(-1)
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    na, nb = np.isnan(a), np.isnan(b)
    nanmask_equal = bool(np.array_equal(na, nb))
    both = ~(na | nb)
    bitdiffs = int(np.count_nonzero(ua[both] != ub[both]))
    zm = a == 0.0
    signzerodiffs = int(np.count_nonzero(np.signbit(a[zm]) != np.signbit(b[zm])))
    if both.any():
        maxabs = float(np.abs(a[both].astype(np.float64)
                              - b[both].astype(np.float64)).max())
    else:
        maxabs = 0.0
    ok = (nanmask_equal and bitdiffs == 0 and signzerodiffs == 0
          and maxabs == 0.0)
    assert ok, (
        "%s is NOT byte-exact vs calib_jungfrau_reference: maxabsdiff=%r "
        "bitdiffs=%d signzerodiffs=%d nanmask_equal=%s"
        % (what, maxabs, bitdiffs, signzerodiffs, nanmask_equal))
    return ("%s maxabsdiff=%r bitdiffs=%d signzerodiffs=%d nanmask_equal=%s "
            "negzero=%d" % (what, maxabs, bitdiffs, signzerodiffs,
                            nanmask_equal,
                            int(np.count_nonzero(zm & np.signbit(a)))))


def _jf_synthetic(seed=20260806):
    """A synthetic jungfrau that exercises every lane of the hybrid kernel.

    Row bands per segment (128 rows each) give the tile classifier something to
    classify at ``tile_rows`` 128 AND 512: band 0 is PURE gain code 0 (case A),
    band 1 is ~0.4% code 1 (the sparse gather), band 2 is ~80% code 1 (the dense
    two-plane blend), band 3 is ~30%.  Stage 2 (``gbits == 3``) and the BAD code
    (``gbits == 2``) are planted in the three non-pure bands so both the dense
    and the sparse tiles take a gather fixup.

    The gain is NEGATIVE for stages 1 and 2 (as the real constants are), which
    is what makes a masked pixel render as ``-0.0`` rather than ``+0.0``, and a
    few pixels have an EXACTLY zero gain so the protected divide's zero factor
    is exercised too.

    Returns ``(raw, pedestals, pixel_gain, pixel_offset, mask, code)``.
    """
    rng = np.random.default_rng(seed)
    nseg, nr, nc = _JF_NSEG, _JF_ROWS, _JF_COLS
    adc = rng.integers(0, 0x4000, size=(nseg, nr, nc), dtype=np.uint16)
    code = np.zeros((nseg, nr, nc), dtype=np.uint16)
    for s in range(nseg):
        for band0, frac in ((128, 0.004), (256, 0.80), (384, 0.30)):
            band = code[s, band0:band0 + 128]
            band[rng.random(band.shape) < frac] = 1
        for band0 in (128, 256, 384):
            rr = rng.integers(band0, band0 + 128, size=64)
            cc = rng.integers(0, nc, size=64)
            code[s, rr[:32], cc[:32]] = 3     # stage 2
            code[s, rr[32:], cc[32:]] = 2     # the BAD code (no gain stage)
    raw = (adc | (code.astype(np.uint32) << 14).astype(np.uint16))
    assert raw.dtype == np.uint16

    ped = rng.uniform(50.0, 4000.0, size=(3, nseg, nr, nc)).astype(np.float32)
    off = rng.uniform(-30.0, 30.0, size=(3, nseg, nr, nc)).astype(np.float32)
    gain = np.empty((3, nseg, nr, nc), dtype=np.float32)
    gain[0] = rng.uniform(0.5, 50.0, size=(nseg, nr, nc))
    gain[1] = -rng.uniform(0.5, 50.0, size=(nseg, nr, nc))
    gain[2] = -rng.uniform(0.5, 50.0, size=(nseg, nr, nc))
    gain[0, 0, 3, :16] = 0.0          # protected divide -> factor exactly 0
    gain[1, 0, 300, :16] = 0.0

    mask = (rng.random((nseg, nr, nc)) > 0.05).astype(np.uint8)
    for want in (2, 3):               # mask HALF the bad-code / stage-2 pixels
        ij = np.argwhere(code == want)
        assert ij.shape[0] > 0, want
        mask[tuple(ij[::2].T)] = 0
    return raw, ped, gain, off, mask, code


def test_gather_fixup_honours_a_noncontiguous_out_buffer():
    """A deliberately NON-CONTIGUOUS ``out=`` still gets every gathered pixel.

    The sparse fixup scatters into ``t.reshape(-1)``.  On a non-contiguous
    ``t`` numpy's ``reshape`` cannot return a view, so it returns a COPY -- the
    scatter then lands in a temporary, is discarded, and every residual pixel
    silently keeps its stage-0 base value.  No exception, no warning.  The
    kernel guards this with ``t.flags["C_CONTIGUOUS"]`` and an
    ``np.unravel_index`` fallback; this test is what makes that guard
    load-bearing, because the campaign bit gate's own output array is always
    contiguous and could never catch its removal.

    NOT EVERY non-contiguous buffer discriminates, and getting this wrong is
    easy: ``big[:, :, ::2]`` is non-contiguous, yet it is UNIFORMLY strided, so
    ``reshape(-1)`` happily returns a VIEW of it and the unguarded gather would
    still land in the right place.  The buffers below are therefore checked with
    ``np.shares_memory(t.reshape(-1), t)`` to be ones on which ``reshape``
    genuinely has to COPY -- a row-stepped view, and a column-prefix of a wider
    array.
    """
    from pscalib.apply import _fastcalib as fc
    from pscalib.apply.jungfrau import (calib_jungfrau,
                                        calib_jungfrau_reference)

    raw, ped, gain, off, mask, code = _jf_synthetic()
    nseg, nrows, ncols = raw.shape
    for want, name in ((0, "G0"), (1, "stage 1"), (3, "stage 2"),
                       (2, "BAD code")):
        assert int(np.count_nonzero(code == want)) > 0, name
    ref = calib_jungfrau_reference(raw, ped, gain, pixel_offset=off, mask=mask)

    fc.memo_clear()
    ctg = np.empty(raw.shape, dtype=np.float32)
    got_c = calib_jungfrau(raw, ped, gain, pixel_offset=off, mask=mask, out=ctg)
    assert got_c is ctg, "out= was silently replaced by a fresh allocation"
    print("[jf-kernel] " + _jf_bits_equal(got_c, ref, "contiguous out="))
    assert fc.LAST_CALL["n_gathered"] > 0, (
        "no pixel took the sparse gather, so this test could not discriminate "
        "the contiguity guard: %r" % (fc.LAST_CALL,))

    POISON = np.float32(-98765.0)
    # (i) row-stepped: rows of ``big`` are twice as far apart as the view's own
    #     row length, so the view is not uniformly strided.
    big_r = np.full((nseg, 2 * nrows, ncols), POISON, dtype=np.float32)
    # (ii) column-prefix of a WIDER array: the classic "leading sub-block".
    big_c = np.full((nseg, nrows, ncols + 7), POISON, dtype=np.float32)
    cases = [("row-stepped out=", big_r[:, ::2, :],
              lambda b=big_r: b[:, 1::2, :]),
             ("column-prefix out=", big_c[:, :, :ncols],
              lambda b=big_c: b[:, :, ncols:])]

    for tag, view, untouched in cases:
        assert view.shape == raw.shape and view.dtype == np.float32
        assert not view.flags["C_CONTIGUOUS"], (
            "%s must be non-contiguous or it proves nothing" % tag)
        probe = view[0]                     # the per-segment array the kernel tiles
        assert not np.shares_memory(probe.reshape(-1), probe), (
            "%s: reshape(-1) returned a VIEW, so an unguarded scatter would "
            "still land correctly and this case cannot discriminate the "
            "contiguity guard" % tag)
        got_v = calib_jungfrau(raw, ped, gain, pixel_offset=off, mask=mask,
                               out=view)
        assert got_v is view
        print("[jf-kernel] " + _jf_bits_equal(got_v, ref, tag))
        assert np.array_equal(np.ascontiguousarray(got_v).view(np.uint32),
                              ctg.view(np.uint32)), \
            "%s differs bit-for-bit from the contiguous run" % tag
        assert np.all(untouched() == POISON), (
            "%s: the kernel wrote outside the view it was handed" % tag)

    # ...and it stays exact for every tiling / threshold, including
    # dense-on-every-tile (0.0) and gather-on-every-tile (1e9).
    view = big_r[:, ::2, :]
    seen = set()
    ngath = 0
    for tile_rows in (128, 512):
        for dense_frac in (0.0, 0.60, 1e9):
            view[...] = POISON
            g = fc.calib_jungfrau_fast(raw, ped, gain, pixel_offset=off,
                                       mask=mask, out=view,
                                       tile_rows=tile_rows,
                                       dense_frac=dense_frac)
            _jf_bits_equal(g, ref, "strided out= tile_rows=%d dense_frac=%g"
                                   % (tile_rows, dense_frac))
            for k in ("A_pure_g0", "B_dense_blend", "C_sparse_gather"):
                if fc.LAST_CALL[k]:
                    seen.add(k)
            ngath += int(fc.LAST_CALL["n_gathered"])
    assert seen == {"A_pure_g0", "B_dense_blend", "C_sparse_gather"}, seen
    assert ngath > 0
    print("[jf-kernel] strided out= byte-exact at tile_rows in {128,512} x "
          "dense_frac in {0.0,0.60,1e9}; all three tile cases exercised "
          "(%s), %d pixels scattered" % (",".join(sorted(seen)), ngath))
    fc.memo_clear()


def test_gfm_fold_is_memoised_on_the_mask_identity():
    """Change ONLY the mask and the fold must be re-derived, not reused.

    ``gfm = gfac * mask`` is a function of BOTH the gain and the mask.  The
    module's memo is keyed on the ``id()`` of its sources (eviction-safe, with
    weakrefs); keyed on the gain alone, this sequence -- same gain, mask A then
    mask B -- would hand mask A's fold to mask B's event and the mask would
    simply be wrong, with no error anywhere.  Both masks are kept ALIVE for the
    whole test so that a pass cannot be an id-recycling accident.
    """
    from pscalib.apply import _fastcalib as fc
    from pscalib.apply.jungfrau import (calib_jungfrau,
                                        calib_jungfrau_reference)

    raw, ped, gain, off, mask_a, _code = _jf_synthetic()
    rng = np.random.default_rng(31337)
    mask_b = mask_a.copy()
    mask_b[rng.random(mask_b.shape) < 0.10] = 0
    mask_b[mask_a == 0] = 1                    # differs in BOTH directions
    assert not np.array_equal(mask_a, mask_b)

    ref_a = calib_jungfrau_reference(raw, ped, gain, pixel_offset=off,
                                     mask=mask_a)
    ref_b = calib_jungfrau_reference(raw, ped, gain, pixel_offset=off,
                                     mask=mask_b)
    assert not np.array_equal(ref_a.view(np.uint32), ref_b.view(np.uint32)), (
        "the two masks give the same output, so this test is vacuous")

    fc.memo_clear()
    got_a1 = calib_jungfrau(raw, ped, gain, pixel_offset=off, mask=mask_a)
    print("[jf-kernel] " + _jf_bits_equal(got_a1, ref_a, "mask A (first)"))
    assert fc.LAST_CALL["mask_folded"] is True, fc.LAST_CALL
    size_a = fc.memo_size()

    # SAME gain object, DIFFERENT mask object -- the whole point.
    got_b = calib_jungfrau(raw, ped, gain, pixel_offset=off, mask=mask_b)
    print("[jf-kernel] " + _jf_bits_equal(got_b, ref_b, "mask B (same gain)"))
    assert fc.LAST_CALL["mask_folded"] is True, fc.LAST_CALL
    assert fc.memo_size() > size_a, (
        "no new memo entry appeared for mask B -- the fold was reused, which "
        "means it is not keyed on the mask (memo_size %d -> %d)"
        % (size_a, fc.memo_size()))

    # ...and back to A: served entirely from the memo (no recompute) and
    # bit-identical to the first A call.
    miss_before = fc.memo_stats()["miss"]
    got_a2 = calib_jungfrau(raw, ped, gain, pixel_offset=off, mask=mask_a)
    assert fc.memo_stats()["miss"] == miss_before, (
        "mask A's constants were re-derived on the third call: %r"
        % (fc.memo_stats(),))
    assert np.array_equal(got_a2.view(np.uint32), got_a1.view(np.uint32))
    print("[jf-kernel] " + _jf_bits_equal(got_a2, ref_a, "mask A (again)"))
    assert mask_a is not None and mask_b is not None   # both still alive
    print("[jf-kernel] gfm is keyed on (pixel_gain, mask): A -> B -> A is "
          "byte-exact each time, %d memo entries, misses %d -> %d on the "
          "repeat" % (fc.memo_size(), miss_before, fc.memo_stats()["miss"]))
    fc.memo_clear()


def test_mask_fold_guard_is_fail_closed():
    """When ``fold_is_exact`` says no, the kernel falls back and stays exact.

    Four cases: a clean 0/1 mask (the DISCRIMINATOR -- the fold must actually be
    taken, or the three poison cases below prove nothing), a mask poisoned with
    a 0.5, a mask carrying a ``-0.0`` on a BAD-code pixel, and a pedestal past
    the guard's magnitude ceiling.  Each poison must (a) turn the fold OFF,
    (b) name the conjunct that declined it, (c) NOT raise, and (d) still be
    byte-exact against the reference computed from the SAME poisoned inputs.
    """
    from pscalib.apply import _fastcalib as fc
    from pscalib.apply.jungfrau import (calib_jungfrau,
                                        calib_jungfrau_reference)

    raw, ped, gain, off, mask, code = _jf_synthetic()

    # (a) DISCRIMINATOR: the clean uint8 0/1 mask IS folded.
    fc.memo_clear()
    got = calib_jungfrau(raw, ped, gain, pixel_offset=off, mask=mask)
    ref = calib_jungfrau_reference(raw, ped, gain, pixel_offset=off, mask=mask)
    assert fc.LAST_CALL["mask_folded"] is True, fc.LAST_CALL
    print("[jf-kernel] " + _jf_bits_equal(got, ref, "clean mask (folded)")
          + " fold_reason=%r" % (fc.LAST_CALL["fold_reason"],))

    def _poisoned(tag, ped_p, mask_p, want_in_reason):
        fc.memo_clear()
        g = calib_jungfrau(raw, ped_p, gain, pixel_offset=off, mask=mask_p)
        r = calib_jungfrau_reference(raw, ped_p, gain, pixel_offset=off,
                                     mask=mask_p)
        assert fc.LAST_CALL["mask_folded"] is False, (
            "%s: the fold was taken anyway -- the guard is not fail-closed "
            "(%r)" % (tag, fc.LAST_CALL))
        assert want_in_reason in fc.LAST_CALL["fold_reason"], (
            "%s: expected the reason to name %r, got %r"
            % (tag, want_in_reason, fc.LAST_CALL["fold_reason"]))
        print("[jf-kernel] " + _jf_bits_equal(g, r, tag)
              + " fold_reason=%r" % (fc.LAST_CALL["fold_reason"],))
        return g, r

    # (b) a mask value that is neither 0 nor 1.  NOTE that 0.5 is a POWER OF
    # TWO, so ``(x*g)*0.5 == x*(g*0.5)`` exactly and folding it anyway would
    # NOT have changed a bit -- the guard still has to decline it (the fold is
    # only ever taken when it is a theorem, never when it merely happens to
    # work), and a value that is NOT a power of two is added right after so the
    # case is discriminating as well as correct.
    mask_half = mask.astype(np.float32)
    mask_half[0, 5, 7] = np.float32(0.5)
    _poisoned("mask with a 0.5 (fold declined)", ped, mask_half,
              "exactly 0 or 1")

    mask_03 = mask.astype(np.float32)
    mask_03[0, 6, :] = np.float32(0.3)      # a whole row, not a power of two
    _poisoned("mask with a 0.3 row (fold declined)", ped, mask_03,
              "exactly 0 or 1")

    # (c) a NEGATIVE ZERO on a BAD-code pixel.  This is the value the folded
    # gather could not reproduce: the reference computes (adc - 0.0) * 0.0 ==
    # +0.0 and then multiplies by the mask, giving -0.0, while a folded gather
    # -- which has no gain plane to fold the mask into for the bad lane -- would
    # leave +0.0.  max|diff| is 0 for that pixel either way; only the sign of
    # the zero tells them apart, so it is asserted directly.
    bad = np.argwhere(code == 2)
    bs, br, bc = (int(x) for x in bad[0])
    mask_neg = mask.astype(np.float32)
    mask_neg[bs, br, bc] = np.float32(-0.0)
    g, r = _poisoned("mask with a -0.0 on a BAD-code pixel (fold declined)",
                     ped, mask_neg, "negative zero")
    assert r[bs, br, bc] == 0.0 and bool(np.signbit(r[bs, br, bc])), (
        "the reference did not produce -0.0 at the poisoned bad-code pixel, "
        "so this case is not discriminating: %r" % (r[bs, br, bc],))
    assert bool(np.signbit(g[bs, br, bc])), (
        "the fallback lost the sign of the zero at the -0.0-masked bad pixel")
    print("[jf-kernel] bad-code pixel (%d,%d,%d) renders -0.0 through the "
          "unfolded fallback, signbit preserved" % (bs, br, bc))

    # (d) a pedestal past the guard's magnitude ceiling.
    ped_big = ped.copy()
    ped_big[0, 0, 0, 0] = np.float32(1e20)
    _poisoned("pedestal 1e20 > FOLD_LIM (fold declined)", ped_big, mask,
              "is not below lim")

    fc.memo_clear()


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-000 acceptance: scaffold pscalib + migrate jungfrau calib + image")
    print("=" * 72)

    # (c) offline import purity always runs (no psana needed)
    test_offline_import_purity_in_proc()
    print("[ok] offline import purity, extended forbidden set (in-proc)")
    test_offline_import_purity_subprocess()
    print("[ok] offline import purity (subprocess, no snapshot)")

    # the fast kernel's three silent-failure invariants (offline, numpy only)
    test_gather_fixup_honours_a_noncontiguous_out_buffer()
    print("[ok] sparse gather honours a NON-contiguous out= buffer "
          "(byte-exact, contiguity guard load-bearing)")
    test_gfm_fold_is_memoised_on_the_mask_identity()
    print("[ok] the gfm fold is memoised on (pixel_gain, mask): changing only "
          "the mask re-derives it (byte-exact)")
    test_mask_fold_guard_is_fail_closed()
    print("[ok] fold_is_exact is fail-closed: a poisoned mask / pedestal "
          "falls back to the unfolded path (byte-exact)")

    if not _have_psana():
        skip("us000_psana_oracle_gates",
             "psana not importable -- the snapshot / byte-exact / "
             "non-regression oracle checks did NOT run. Source psconda.sh on "
             "sdfiana025 (and PREPEND, never replace, PYTHONPATH).")
        print("\nUS-000 offline-purity checks PASSED (psana-dependent checks "
              "SKIPPED -- see the ##SKIP## line above; this run proves nothing "
              "about byte-exactness)")
        return

    tmp = tempfile.mkdtemp(prefix="pscalib_us000_")
    try:
        # reload byte-exact vs psana (the calib half)
        test_snapshot_reload_byte_exact(out_dir=os.path.join(tmp, "reload"))
        print("[ok] reload byte-exact vs psana _calibconst (np.array_equal)")

        # (b) render byte-exact vs psana for an evt.timestamp() event
        snap_dir = test_render_byte_exact(out_dir=os.path.join(tmp, "render"))
        print("[ok] (b) offline render calib + image byte-exact vs psana "
              "(max|diff| == 0)")

        # (c) full offline apply in a fresh interpreter stays clean
        test_offline_import_purity_subprocess(snapshot_dir=snap_dir)
        print("[ok] (c) offline apply import purity (subprocess, full render)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL US-000 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
