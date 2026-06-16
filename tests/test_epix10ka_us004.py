#!/usr/bin/env python3
"""US-004 acceptance test: epix10ka pure-numpy apply plugin (7 gain ranges).

Verifies the US-004 acceptance criteria for the reference epix10ka dataset
(exp=ued1010667, run=177, dir=/sdf/data/lcls/ds/prj/public01/xtc,
det='epixquad', class epix10ka_raw_2_0_1):

  (a) BYTE-EXACT GATE.  For ONE event whose 64-bit timestamp comes from psana
      itself (``evt.timestamp()``), pscalib's calib ``(4,352,384) f32`` equals
      ``det.raw.calib(evt)`` with ``max|diff| == 0``.  The gain range is decoded
      from the per-ASIC ``trbit`` / ``asicPixelConfig`` (psdata ``seg_configs``,
      US-003) OR-ed with the per-event data gain bit -- NOT from the calib DB.
      Reference shapes: raw ``(4,352,384) uint16``; pedestals / pixel_gain
      ``(7,4,352,384) f32``.

  (b) REGISTRY DISPATCH.  The epix10ka plugin registers into the SAME
      registry/dispatch as jungfrau via the thin seam
      ``plugin(raw, constants, config=None) -> calib``; the registry resolves
      both the bare ``"epix10ka"`` type and the psana class name
      ``"epix10ka_raw_2_0_1"`` to the plugin, and ``pscalib.calib(...)``
      produces the same byte-exact result.

  (c) IMPORT PURITY.  The apply path imports only numpy --
      ('psana','mpi4py','h5py','dgram','pymongo') absent after the apply (both
      in-proc and in a fresh interpreter).

  (d) CROSS-PROVIDER.  The same calib is produced whether the mask comes from
      the snapshot's cached ``det.raw._mask()`` or is derived from
      ``pixel_status`` via ``mask_from_pixel_status`` (the BYO / web path).

The byte-exact checks need the PRODUCTION psana env (psconda.sh) to GENERATE
ground truth + snapshot constants -- run on sdfiana025 via
``run_tests.sh tests/test_epix10ka_us004.py``.  The offline import-purity and
registry-wiring checks run without psana.
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

# Reference dataset -- lives in the TEST, never in the library.
EXP = "ued1010667"
RUN = 177
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "epixquad"
SEGS = [0, 1, 2, 3]

# Reference shapes/dtypes (HANDOFF datasets section).
RAW_SHAPE = (4, 352, 384)
CALIB_SHAPE = (4, 352, 384)
CONS_SHAPE = (7, 4, 352, 384)          # leading 7 = epix10ka gain ranges

# The forbidden set (shared with pscalib._purity).
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


# --------------------------------------------------------------------------
# (b)+(c) registry wiring + offline import purity -- no psana needed
# --------------------------------------------------------------------------
def test_registry_wiring_offline():
    """The epix10ka plugin is registered alongside jungfrau under one dispatch,
    and the apply leaf math is reachable -- all numpy-only."""
    import pscalib
    import pscalib.registry as reg

    types = reg.registered_types()
    assert "epix10ka" in types, types
    assert "jungfrau" in types, types

    # both the bare family name and the psana drp class name resolve
    assert reg.detector_type_of("epix10ka") == "epix10ka"
    assert reg.detector_type_of("epix10ka_raw_2_0_1") == "epix10ka"
    assert reg.detector_type_of("epixquad") == "epix10ka"
    assert reg.detector_type_of("jungfrau_raw_0_1_0") == "jungfrau"
    assert reg.get_plugin("epix10ka_raw_2_0_1") is reg.plugin_epix10ka
    assert reg.get_plugin("jungfrau") is reg.plugin_jungfrau

    # the public surface exposes the unified entry + the leaf
    assert pscalib.calib is reg.calib
    assert callable(pscalib.calib_epix10ka)
    assert callable(pscalib.mask_from_pixel_status)

    # epix plugin REQUIRES the config object (load-bearing dependency)
    try:
        reg.plugin_epix10ka(np.zeros(RAW_SHAPE, np.uint16),
                            {"pedestals": np.zeros(CONS_SHAPE, np.float32),
                             "pixel_gain": np.ones(CONS_SHAPE, np.float32)},
                            config=None)
        raise AssertionError("epix10ka plugin must refuse config=None")
    except ValueError:
        pass
    print("[ok] registry wiring: epix10ka + jungfrau share one dispatch")


def test_offline_import_purity_in_proc():
    """Importing pscalib (incl. the epix10ka apply + registry) must not pull in
    any framework."""
    import pscalib
    _ = (pscalib.calib_epix10ka, pscalib.calib, pscalib.registry)
    assert pscalib.FORBIDDEN_MODULES == FORBIDDEN, pscalib.FORBIDDEN_MODULES
    pscalib.assert_no_framework_imports()
    for m in FORBIDDEN:
        assert m not in sys.modules, f"{m} leaked into sys.modules on import"
    print("[ok] offline import purity (in-proc)")


def test_offline_apply_purity_subprocess():
    """In a FRESH interpreter: import pscalib + run the epix10ka apply on
    synthetic arrays through the registry.  None of the forbidden modules may
    appear; numpy must.  No psana, no psdata DataSource."""
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
        "st = np.zeros((7, 4, 352, 384), np.uint64)\n"
        "raw = np.zeros((4, 352, 384), np.uint16)\n"
        "cfg = {i: _Seg(_Ns(np.zeros(4, np.uint8),\n"
        "                   np.zeros((4, 176, 192), np.uint8))) for i in range(4)}\n"
        "cons = {'pedestals': ped, 'pixel_gain': gain, 'pixel_status': st}\n"
        "out = pscalib.calib('epix10ka_raw_2_0_1', raw, cons, config=cfg)\n"
        "assert out.shape == (4, 352, 384) and out.dtype == np.float32, (out.shape, out.dtype)\n"
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
    print("[ok] offline apply import purity (subprocess)")


# --------------------------------------------------------------------------
# (a)+(d) byte-exact gate + cross-provider mask -- needs psana + psdata
# --------------------------------------------------------------------------
def test_epix10ka_byte_exact(out_dir):
    """Snapshot the reference epixquad run (psana), read raw + config offline
    (psdata), apply in pure numpy through the registry, and assert byte-identical
    to ``det.raw.calib(evt)`` for one ``evt.timestamp()`` event."""
    import pscalib
    import pscalib.registry as reg
    import pscalib.providers.snapshot as ps_snap

    # --- regenerate psana ground truth ourselves ------------------------
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    evt = next(myrun.events())
    ts64 = evt.timestamp() if callable(getattr(evt, "timestamp", None)) \
        else evt.timestamp
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_calib = np.asarray(det.raw.calib(evt))
    assert gt_raw.shape == RAW_SHAPE and gt_raw.dtype == np.uint16, gt_raw.shape
    assert gt_calib.shape == CALIB_SHAPE and gt_calib.dtype == np.float32
    det_type = type(det.raw).__name__       # e.g. epix10ka_raw_2_0_1
    print(f"[gt] ts={ts64} raw {gt_raw.shape} calib {gt_calib.shape} "
          f"det_type={det_type!r}")

    # --- one-time snapshot of constants (captures the default mask) ------
    snap_dir = ps_snap.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                      out_dir=out_dir)
    snap = ps_snap.load_snapshot(snap_dir)
    assert snap.pedestals.shape == CONS_SHAPE, snap.pedestals.shape
    assert snap.pixel_gain.shape == CONS_SHAPE, snap.pixel_gain.shape
    print(f"[snap] {snap!r} pedestals {snap.pedestals.shape} "
          f"mask {None if snap.mask is None else snap.mask.shape}")

    # --- read raw + the per-ASIC Configure object fully offline (psdata) -
    import psdata
    run = psdata.open(exp=EXP, run=RUN, dir=DIR)
    seg_cfg = run.seg_configs(DET)          # {seg: ns(ns.config.trbit/apc)}
    assert sorted(seg_cfg) == SEGS, sorted(seg_cfg)
    # pull the SAME event by its psana timestamp so raw matches gt_raw
    pevt = run.read_event(int(ts64))
    raw = pevt.stack(DET, field="raw", alg="raw")
    assert raw is not None and raw.shape == RAW_SHAPE, \
        (None if raw is None else raw.shape)
    assert np.array_equal(raw, gt_raw), "psdata raw != psana raw (wrong event?)"
    print(f"[psdata] raw {raw.shape} matches psana; seg_configs {sorted(seg_cfg)}")

    # --- pure-numpy apply through the REGISTRY (the thin seam) -----------
    constants = snap                        # CalibSnapshot is a constants mapping
    my_calib = pscalib.calib(det_type, raw, constants, config=seg_cfg)
    assert my_calib.shape == CALIB_SHAPE, my_calib.shape
    assert my_calib.dtype == np.float32, my_calib.dtype
    d = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(gt_calib))
    assert np.array_equal(my_calib, gt_calib), \
        f"calib not byte-exact: max|diff|={d.max()}"
    print(f"[byte-exact] registry calib {my_calib.shape} {my_calib.dtype} "
          f"max|diff|={d.max()} array_equal=True")

    # the leaf function directly == the registry result
    leaf = pscalib.calib_epix10ka(raw, snap.pedestals, snap.pixel_gain, seg_cfg,
                                  mask=snap.mask)
    assert np.array_equal(leaf, my_calib), "leaf != registry dispatch"

    # (d) CROSS-PROVIDER: mask from snapshot vs derived from pixel_status
    status = snap.array("pixel_status")
    assert status is not None and status.shape == CONS_SHAPE, \
        (None if status is None else status.shape)
    derived_mask = pscalib.mask_from_pixel_status(status)
    assert np.array_equal(derived_mask, np.asarray(snap.mask)), \
        "derived status mask != snapshot's cached _mask()"
    # apply with the derived mask (the BYO / web path) -> same byte-exact calib
    byo_constants = {"pedestals": snap.pedestals,
                     "pixel_gain": snap.pixel_gain,
                     "pixel_status": status}
    byo_calib = pscalib.calib(det_type, raw, byo_constants, config=seg_cfg)
    assert np.array_equal(byo_calib, gt_calib), \
        "BYO (mask-from-status) calib not byte-exact"
    print("[cross-provider] snapshot-mask and pixel_status-derived-mask give "
          "the same byte-exact calib")

    # (c) apply path stayed framework-free even after the live read above:
    #     the apply itself only touched numpy.  (psana/psdata were imported by
    #     THIS process to make ground truth; the fresh-interpreter subprocess
    #     test is the authoritative purity gate.)
    return snap_dir


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-004 acceptance: epix10ka pure-numpy apply plugin (7 gain ranges)")
    print("=" * 72)

    # offline checks always run (no psana needed)
    test_registry_wiring_offline()
    test_offline_import_purity_in_proc()
    test_offline_apply_purity_subprocess()

    if not _have_psana():
        print("\n[skip] psana not importable -- byte-exact gate skipped. "
              "Source psconda.sh on sdfiana025.")
        print("\nUS-004 offline checks PASSED (psana-dependent gate skipped)")
        return
    if not _have_psdata():
        print("\n[skip] psdata not importable -- pscalib depends on it. "
              "Put psdata/src on PYTHONPATH (run_tests.sh does).")
        return

    tmp = tempfile.mkdtemp(prefix="pscalib_us004_")
    try:
        test_epix10ka_byte_exact(out_dir=tmp)
        print("[ok] (a) byte-exact gate calib (4,352,384) f32 vs psana "
              "(max|diff| == 0)")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL US-004 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
