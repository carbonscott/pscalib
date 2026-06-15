#!/usr/bin/env python3
"""US-000 acceptance test: scaffold pscalib + migrate psdata's jungfrau calib
+ hdr into it (non-regressing).

Verifies the US-000 acceptance criteria for the reference Jungfrau dataset
(exp=mfx100848724, run=51, dir=/sdf/data/lcls/ds/prj/public01/xtc, det=jungfrau):

  (a) NON-REGRESSION -- snapshot byte-identical to psdata's.  A snapshot
      produced by pscalib's lifted ``snapshot_calib`` has every ``.npy`` array
      AND the ``manifest.json`` sha1-identical to one produced by psdata's
      current ``snapshot_calib`` for the same (det, run).

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

  (d) NO DUPLICATE-CANONICAL HAZARD -- psdata's calib/ + hdr/ are a re-export
      shim of pscalib (see test_no_drift_us000.py).

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

# Per-acceptance expected shapes/dtypes of the named HDR constants.
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
    _ = (pscalib.HDRImager, pscalib.calib_jungfrau, pscalib.assemble_image,
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
            "from pscalib import load_snapshot, HDRImager; "
            f"snap=load_snapshot({snapshot_dir!r}); "
            "im=HDRImager(snap, derive_geometry_if_missing=False); "
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
# (a) non-regression: pscalib snapshot == psdata snapshot, sha1-identical
# --------------------------------------------------------------------------
def test_snapshot_matches_psdata(out_dir):
    """pscalib's lifted snapshot_calib must produce a snapshot sha1-identical
    (every .npy AND manifest.json) to psdata's current snapshot_calib for the
    same (det, run)."""
    import pscalib.providers.snapshot as ps_snap

    # pscalib snapshot
    ps_dir = ps_snap.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                    out_dir=os.path.join(out_dir, "pscalib"))
    # psdata snapshot (the reference). psdata is a dependency; import it.
    import psdata.calib as pd_calib
    pd_dir = pd_calib.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                     out_dir=os.path.join(out_dir, "psdata"))

    ps_h = _sha1_dir(ps_dir)
    pd_h = _sha1_dir(pd_dir)
    assert set(ps_h) == set(pd_h), (
        f"file sets differ: pscalib={sorted(ps_h)} psdata={sorted(pd_h)}")
    for name in sorted(ps_h):
        assert ps_h[name] == pd_h[name], (
            f"sha1 drift for {name!r}: pscalib={ps_h[name]} psdata={pd_h[name]}")
    # explicit: manifest.json must be byte-identical (catches a schema string
    # or field-ordering drift).
    assert ps_h["manifest.json"] == pd_h["manifest.json"], "manifest.json drift"
    print(f"[non-regression] {len(ps_h)} files sha1-identical "
          f"(incl. manifest.json): {sorted(ps_h)}")
    return ps_dir


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
    imager = pscalib.HDRImager(snap, derive_geometry_if_missing=False)
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
def main():
    print("=" * 72)
    print("US-000 acceptance: scaffold pscalib + migrate jungfrau calib + hdr")
    print("=" * 72)

    # (c) offline import purity always runs (no psana needed)
    test_offline_import_purity_in_proc()
    print("[ok] offline import purity, extended forbidden set (in-proc)")
    test_offline_import_purity_subprocess()
    print("[ok] offline import purity (subprocess, no snapshot)")

    if not _have_psana():
        print("\n[skip] psana not importable -- snapshot/byte-exact/"
              "non-regression checks skipped. Source psconda.sh on sdfiana025.")
        print("\nUS-000 offline-purity checks PASSED (psana-dependent checks "
              "skipped)")
        return

    tmp = tempfile.mkdtemp(prefix="pscalib_us000_")
    try:
        # (a) snapshot non-regression vs psdata
        test_snapshot_matches_psdata(out_dir=os.path.join(tmp, "regr"))
        print("[ok] (a) snapshot sha1-identical to psdata's (non-regression)")

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
