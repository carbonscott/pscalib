#!/usr/bin/env python3
"""US-005 acceptance test: unified public API + multi-detector coverage gate.

Verifies the US-005 acceptance criteria:

  (1) PUBLIC SURFACE.  ``pscalib.calib(raw, constants, config=None)`` dispatches
      to the right detector plugin via the registry, inferring the detector type
      from the constants alone (no ``det_type`` argument).  Constants come from
      any provider -- snapshot (US-000), web (US-001), or a BYO dict -- behind
      one uniform ``pscalib.model.Constants`` contract, with validity
      enforcement (US-002) applied when a ``run`` is given.  The legacy explicit
      form ``pscalib.calib(det_type, raw, constants, config)`` (US-004) still
      works -- one entry point, one registry dispatch.

  (2) MULTI-DETECTOR COVERAGE GATE, in ONE harness that regenerates psana
      ground truth itself:
        * jungfrau (mfx100848724/r51): calib (32,512,1024) f32 AND
          image (4216,4432) f32 both == det.raw.calib/image(evt), max|diff|==0;
        * epix10ka (ued1010667/r177, det='epixquad'): calib (4,352,384) f32
          == det.raw.calib(evt), max|diff|==0;
      both driven through the inferred public surface ``pscalib.calib(raw,
      constants, config=...)``.
        * CROSS-PROVIDER: the same jungfrau IMAGE is produced whether the
          constants came from the snapshot provider or the web provider.

  (3) IMPORT PURITY.  Importing pscalib pulls in only numpy; requests+bson only
      under the 'web' extra; psana never.  After the whole import +
      offline/inferred apply, ('psana','mpi4py','h5py','dgram','pymongo') are
      absent (in-proc AND in a fresh interpreter).

  (4) GAP NOTE (not blocking): the off-site psextapi endpoint is not routable
      from sdfiana025; the on-site psdmint provider gate (US-001) is the real
      correctness bar.  The cross-provider check uses the on-site web service.

The byte-exact / cross-provider checks need the PRODUCTION psana env (psconda.sh)
to GENERATE ground truth + snapshot constants, and on-site network for the web
fetch -- run on sdfiana025 via ``run_tests.sh tests/test_api_us005.py``.  The
offline public-surface + purity checks run without psana.
"""

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

# Reference datasets -- live in the TEST, never in the library.
JF = dict(exp="mfx100848724", run=51,
          dir="/sdf/data/lcls/ds/prj/public01/xtc", det="jungfrau")
EPIX = dict(exp="ued1010667", run=177,
            dir="/sdf/data/lcls/ds/prj/public01/xtc", det="epixquad",
            segs=[0, 1, 2, 3])

JF_CALIB_SHAPE = (32, 512, 1024)
EPIX_CALIB_SHAPE = (4, 352, 384)

# The whole-import forbidden set (extends psdata's three).
FORBIDDEN = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


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


def _ts64(evt):
    return evt.timestamp() if callable(getattr(evt, "timestamp", None)) \
        else evt.timestamp


# ==========================================================================
# (1) PUBLIC SURFACE -- offline, no psana
# ==========================================================================
def test_public_surface_offline():
    """The unified public surface: inferred + explicit calib forms agree, the
    Constants contract is uniform, det_type is inferred from the constants, and
    validity enforcement is wired in -- all numpy-only."""
    import pscalib
    import pscalib.registry as reg

    # the public entry IS the registry dispatch (one entry point)
    assert pscalib.calib is reg.calib
    assert "jungfrau" in reg.registered_types()
    assert "epix10ka" in reg.registered_types()

    # the uniform Constants contract is exposed
    assert hasattr(pscalib, "Constants")
    assert callable(pscalib.detector_type_for_constants)

    # --- a BYO psana-style dict carrying its detector identity (jungfrau) ----
    ped = np.zeros((3, 32, 512, 1024), np.float32)
    gain = np.ones((3, 32, 512, 1024), np.float32)
    raw = np.zeros((32, 512, 1024), np.uint16)
    meta = {"dettype": "jungfrau", "run": 49, "run_end": "end"}
    cons = {"pedestals": (ped, meta), "pixel_gain": (gain, meta)}

    # det_type inferred from the constants alone (no det_type arg)
    assert pscalib.detector_type_for_constants(cons) == "jungfrau"

    # INFERRED form == EXPLICIT (legacy) form, byte-for-byte
    out_inferred = pscalib.calib(raw, cons)
    out_explicit = pscalib.calib("jungfrau", raw, cons)
    assert out_inferred.shape == JF_CALIB_SHAPE
    assert out_inferred.dtype == np.float32
    assert np.array_equal(out_inferred, out_explicit), \
        "inferred and explicit calib forms must agree"

    # the Constants adapter is a uniform, idempotent view over any source
    C = pscalib.Constants.of(cons)
    assert pscalib.Constants.of(C) is C                # idempotent .of()
    assert pscalib.Constants(C).source is cons         # idempotent wrap
    assert C.det_type_hint == "jungfrau"
    assert pscalib.detector_type_for_constants(C) == "jungfrau"
    assert C.array("pedestals") is ped                 # unwraps (arr, meta)
    assert set(C.validities()) == {"pedestals", "pixel_gain"}
    out_wrapped = pscalib.calib(raw, C)
    assert np.array_equal(out_inferred, out_wrapped), \
        "Constants-wrapped source must give the same calib"

    # --- validity enforcement (US-002) wired into the public surface --------
    pscalib.calib(raw, cons, run=51)                   # in range -> silent
    try:
        pscalib.calib(raw, cons, run=10)               # below first-valid -> stale
        raise AssertionError("expected StaleConstantsError for an out-of-range run")
    except pscalib.StaleConstantsError as e:
        assert e.run == 10 and len(e.offenders) == 2
    out_stale = pscalib.calib(raw, cons, run=10, allow_stale=True)   # warn+proceed
    assert np.array_equal(out_stale, out_inferred), \
        "allow_stale must not change the arithmetic, only the refusal"

    # constants with no detector identity -> a clear refusal, not a guess
    try:
        pscalib.calib(raw, {"pedestals": ped, "pixel_gain": gain})
        raise AssertionError("expected ValueError when det_type is unrecoverable")
    except ValueError:
        pass

    # whole import stayed numpy-only
    assert pscalib.FORBIDDEN_MODULES == FORBIDDEN
    pscalib.assert_no_framework_imports()
    for m in FORBIDDEN:
        assert m not in sys.modules, f"{m} leaked into sys.modules"
    assert "requests" not in sys.modules, "import pscalib pulled requests"
    print("[ok] (1) public surface: inferred==explicit, Constants contract, "
          "det_type inference, validity enforcement -- numpy-only")


def test_whole_import_purity_subprocess():
    """In a FRESH interpreter: import pscalib + run the inferred public apply on
    synthetic arrays for BOTH detectors.  Only numpy; none of the forbidden
    modules; no requests (the web extra is opt-in, never on import)."""
    script = (
        "import sys\n"
        "import numpy as np\n"
        "import pscalib\n"
        "assert 'requests' not in sys.modules, 'import pscalib pulled requests'\n"
        # jungfrau via the inferred public surface
        "m = {'dettype': 'jungfrau', 'run': 0, 'run_end': 'end'}\n"
        "ped = np.zeros((3,32,512,1024), np.float32)\n"
        "gain = np.ones((3,32,512,1024), np.float32)\n"
        "raw = np.zeros((32,512,1024), np.uint16)\n"
        "cons = {'pedestals': (ped, m), 'pixel_gain': (gain, m)}\n"
        "out = pscalib.calib(raw, cons)\n"
        "assert out.shape == (32,512,1024) and out.dtype == np.float32\n"
        # epix10ka via the inferred public surface (config required)
        "class _Ns:\n"
        "    def __init__(self, t, a):\n"
        "        self.trbit = t; self.asicPixelConfig = a\n"
        "class _Seg:\n"
        "    def __init__(self, c):\n"
        "        self.config = c\n"
        "me = {'dettype': 'epix10ka', 'run': 0, 'run_end': 'end'}\n"
        "pe = np.zeros((7,4,352,384), np.float32)\n"
        "ge = np.ones((7,4,352,384), np.float32)\n"
        "re = np.zeros((4,352,384), np.uint16)\n"
        "cfg = {i: _Seg(_Ns(np.zeros(4, np.uint8),\n"
        "                   np.zeros((4,176,192), np.uint8))) for i in range(4)}\n"
        "ce = {'pedestals': (pe, me), 'pixel_gain': (ge, me)}\n"
        "oe = pscalib.calib(re, ce, config=cfg)\n"
        "assert oe.shape == (4,352,384) and oe.dtype == np.float32\n"
        "pscalib.assert_no_framework_imports()\n"
        "bad = [x for x in %r if x in sys.modules]\n"
        "assert not bad, bad\n"
        "assert 'numpy' in sys.modules\n"
        "print('CLEAN')\n"
        % (FORBIDDEN,)
    )
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _PKG_PARENT + (os.pathsep + inherited if inherited else "")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout
    print("[ok] (3) whole-import purity (fresh interp, both detectors inferred)")


# ==========================================================================
# (2) MULTI-DETECTOR COVERAGE GATE -- ONE harness, regenerates psana GT itself
# ==========================================================================
def test_jungfrau_gate_and_cross_provider(out_dir):
    """jungfrau byte-exact calib + image through the inferred public surface,
    plus the cross-provider check: the same image whether the constants came
    from the snapshot provider or the on-site web provider."""
    import pscalib
    import pscalib.providers.snapshot as ps_snap
    import pscalib.geometry as pgeo
    from pscalib.image import assemble_image

    from psana import DataSource
    ds = DataSource(exp=JF["exp"], run=JF["run"], dir=JF["dir"])
    myrun = next(ds.runs())
    det = myrun.Detector(JF["det"])
    evt = next(myrun.events())
    ts64 = _ts64(evt)
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_calib = np.asarray(det.raw.calib(evt))
    gt_image = np.asarray(det.raw.image(evt))
    uniqueid = det.raw._uniqueid
    assert gt_calib.shape == JF_CALIB_SHAPE and gt_calib.dtype == np.float32
    print(f"[jf gt] ts={ts64} calib {gt_calib.shape} image {gt_image.shape}")

    # one-time snapshot + cached geometry index maps
    snap_dir = ps_snap.snapshot_calib(out_dir=out_dir, **{k: JF[k] for k in
                                      ("exp", "run", "dir")},
                                      detname=JF["det"])
    pgeo.cache_pixel_indexes_for_snapshot(snap_dir)
    snap = ps_snap.load_snapshot(snap_dir)
    imager = pscalib.Imager(snap, derive_geometry_if_missing=False)

    # --- jungfrau byte-exact through the INFERRED public surface ------------
    my_calib = pscalib.calib(gt_raw, snap)            # det_type inferred
    assert my_calib.shape == JF_CALIB_SHAPE and my_calib.dtype == np.float32
    dcal = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(gt_calib))
    assert np.array_equal(my_calib, gt_calib), \
        f"jungfrau calib not byte-exact: max|diff|={dcal.max()}"
    my_image = imager.image(my_calib)
    di = np.abs(np.nan_to_num(my_image) - np.nan_to_num(gt_image))
    assert np.array_equal(my_image, gt_image), \
        f"jungfrau image not byte-exact: max|diff|={di.max()}"
    print(f"[jf byte-exact] calib {my_calib.shape} max|diff|={dcal.max()} | "
          f"image {my_image.shape} max|diff|={di.max()} (inferred surface)")

    # --- CROSS-PROVIDER: snapshot constants vs on-site web constants --------
    from pscalib.providers import webdb
    web_cons = webdb.get_constants(uniqueid, exp=JF["exp"], run=JF["run"])
    assert web_cons is not None and "pedestals" in web_cons, \
        "web fetch returned nothing (on-site psdmint unreachable?)"
    # web fetch names its own detector type -> inferred dispatch, no det_type
    assert pscalib.detector_type_for_constants(web_cons) == "jungfrau"
    web_calib = pscalib.calib(gt_raw, web_cons)       # det_type inferred from web meta
    assert np.array_equal(web_calib, my_calib), \
        "web-provider calib != snapshot-provider calib (US-001 byte-exactness)"
    # assemble the web-constants calib with the SAME geometry index maps
    web_image = assemble_image(web_calib, imager.ix, imager.iy,
                               rc_tot_max=imager._rc_tot_max)
    assert np.array_equal(web_image, my_image), \
        "cross-provider IMAGE mismatch (snapshot vs web constants)"
    assert np.array_equal(web_image, gt_image), \
        "web-provider image not byte-exact vs psana"
    print("[cross-provider] same jungfrau image from snapshot AND web "
          "constants (both byte-exact vs psana)")

    # the web fetch itself went over HTTP (requests), not the Mongo path:
    assert "requests" in sys.modules, "web fetch did not import requests"
    # NOTE: this process opened a psana DataSource above to GENERATE ground
    # truth, so psana/mpi4py are legitimately in sys.modules here -- an
    # in-process purity check would be meaningless.  The authoritative purity
    # gate is the fresh-interpreter subprocess (test_whole_import_purity_*),
    # exactly as US-001 / US-004 do.
    return snap_dir


def test_epix10ka_gate(out_dir):
    """epix10ka byte-exact calib through the inferred public surface (config
    required), for one evt.timestamp() event."""
    import pscalib
    import pscalib.providers.snapshot as ps_snap

    from psana import DataSource
    ds = DataSource(exp=EPIX["exp"], run=EPIX["run"], dir=EPIX["dir"])
    myrun = next(ds.runs())
    det = myrun.Detector(EPIX["det"])
    evt = next(myrun.events())
    ts64 = _ts64(evt)
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_calib = np.asarray(det.raw.calib(evt))
    assert gt_calib.shape == EPIX_CALIB_SHAPE and gt_calib.dtype == np.float32
    print(f"[epix gt] ts={ts64} calib {gt_calib.shape}")

    snap_dir = ps_snap.snapshot_calib(out_dir=out_dir,
                                      **{k: EPIX[k] for k in ("exp", "run", "dir")},
                                      detname=EPIX["det"])
    snap = ps_snap.load_snapshot(snap_dir)

    # read raw + the per-ASIC Configure object fully offline (psdata)
    import psdata
    run = psdata.open(exp=EPIX["exp"], run=EPIX["run"], dir=EPIX["dir"])
    seg_cfg = run.seg_configs(EPIX["det"])
    assert sorted(seg_cfg) == EPIX["segs"], sorted(seg_cfg)
    pevt = run.read_event(int(ts64))
    raw = pevt.stack(EPIX["det"], field="raw", alg="raw")
    assert raw is not None and np.array_equal(raw, gt_raw), \
        "psdata raw != psana raw (wrong event?)"

    # epix10ka byte-exact through the INFERRED public surface (det_type inferred)
    assert pscalib.detector_type_for_constants(snap) == "epix10ka"
    my_calib = pscalib.calib(raw, snap, config=seg_cfg)
    assert my_calib.shape == EPIX_CALIB_SHAPE and my_calib.dtype == np.float32
    d = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(gt_calib))
    assert np.array_equal(my_calib, gt_calib), \
        f"epix10ka calib not byte-exact: max|diff|={d.max()}"
    print(f"[epix byte-exact] calib {my_calib.shape} max|diff|={d.max()} "
          "(inferred surface)")
    return snap_dir


# ==========================================================================
def main():
    print("=" * 72)
    print("US-005 acceptance: unified public API + multi-detector coverage gate")
    print("=" * 72)

    # offline public-surface + purity checks always run (no psana needed)
    test_public_surface_offline()
    test_whole_import_purity_subprocess()

    if not _have_psana():
        print("\n[skip] psana not importable -- multi-detector byte-exact + "
              "cross-provider gate skipped. Source psconda.sh on sdfiana025.")
        print("\nUS-005 offline checks PASSED (psana-dependent gate skipped)")
        return
    if not _have_psdata():
        print("\n[skip] psdata not importable -- pscalib depends on it. "
              "Put psdata/src on PYTHONPATH (run_tests.sh does).")
        return

    tmp = tempfile.mkdtemp(prefix="pscalib_us005_")
    try:
        # (2) ONE harness regenerates psana GT itself for BOTH detectors
        test_jungfrau_gate_and_cross_provider(out_dir=os.path.join(tmp, "jf"))
        print("[ok] (2a) jungfrau calib (32,512,1024) + image (4216,4432) "
              "byte-exact vs psana (max|diff| == 0), inferred surface")
        test_epix10ka_gate(out_dir=os.path.join(tmp, "epix"))
        print("[ok] (2b) epix10ka calib (4,352,384) byte-exact vs psana "
              "(max|diff| == 0), inferred surface")
        print("[ok] (2c) cross-provider: same jungfrau image from snapshot "
              "AND on-site web constants")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL US-005 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
