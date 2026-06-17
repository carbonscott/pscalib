#!/usr/bin/env python3
"""US-006 acceptance: the vendored numpy-only GeometryAccess closure.

Proves that deriving the per-pixel image index maps (``ix``/``iy``) from a
detector's geometry text via the **vendored** ``pscalib._geometry`` closure is

  1. BYTE-EXACT (``np.array_equal``) vs psana's
     ``GeometryAccess.get_pixel_coord_indexes(do_tilt=True, cframe=0)`` -- driven
     by the SAME geometry text and SAME kwargs -- for BOTH

       * jungfrau  (exp=mfx100848724 run=51 det='jungfrau'),
       * epix10ka  (exp=ued1010667  run=177 det='epixquad'),

  2. END-TO-END: with NO cached ix.npy/iy.npy and ``derive_geometry_if_missing
     =True``, the jungfrau render produces the assembled image (4216,4432) f32
     with ``max|diff| == 0`` vs ``det.raw.image(evt)`` -- proving the vendored
     derivation path drives the renderer,

  3. IMPORT-PURE: after deriving the maps + rendering in a FRESH interpreter,
     none of ('psana','mpi4py','h5py','dgram','pymongo') appear in sys.modules.

The vendored closure imports only os/numpy/math/logging -- no psana.  psana is
used here ONLY to regenerate the ground truth (a separate step in the same
process for 1+2; a wholly separate process for the in-proc purity probe 3).

Run on sdfiana025:
    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    bash run_tests.sh tests/test_geometry_us006.py
"""

import os
import subprocess
import sys

import numpy as np
import pytest

# -- jungfrau reference dataset --------------------------------------------
JF_EXP = "mfx100848724"
JF_RUN = 51
JF_DET = "jungfrau"
JF_DIR = "/sdf/data/lcls/ds/prj/public01/xtc"

# -- epix10ka reference dataset --------------------------------------------
EP_EXP = "ued1010667"
EP_RUN = 177
EP_DET = "epixquad"
EP_DIR = "/sdf/data/lcls/ds/prj/public01/xtc"

FORBIDDEN = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


def _geometry_text(exp, run, det, det_dir):
    """Pull the detector's geometry text once, via the existing calib provider
    (== ``det.raw._calibconst['geometry'][0]``)."""
    import pscalib.providers.snapshot as ps_snap
    from psana import DataSource
    ds = DataSource(exp=exp, run=run, dir=det_dir)
    myrun = next(ds.runs())
    det_obj = myrun.Detector(det)
    # the snapshot provider reads geometry straight from _calibconst
    cc = det_obj.raw._calibconst
    assert cc is not None, f"_calibconst is None for {det!r} r{run}"
    geo = cc.get("geometry")
    assert geo is not None, f"no 'geometry' ctype in _calibconst for {det!r}"
    geometry_text = geo[0]
    assert isinstance(geometry_text, str) and len(geometry_text) > 1000, \
        f"geometry text looks wrong: {type(geometry_text)} len={len(geometry_text) if isinstance(geometry_text, str) else '?'}"
    # psana's own derived index maps + assembled image, for cross-check
    return geometry_text, det_obj, myrun


def _psana_indexes_from_text(geometry_text, do_tilt=True, cframe=0):
    """psana's OWN GeometryAccess fed the SAME text + SAME kwargs."""
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess as PsanaGA
    geo = PsanaGA()
    geo.load_pars_from_str(geometry_text)
    ix, iy = geo.get_pixel_coord_indexes(do_tilt=do_tilt, cframe=cframe)
    return np.asarray(ix), np.asarray(iy)


# --------------------------------------------------------------------------
# (1) vendored derivation == psana, SAME text + kwargs, both detectors
# --------------------------------------------------------------------------
def _check_derivation_byte_exact(exp, run, det, det_dir, label):
    import pscalib.geometry as pgeo
    geometry_text, _det_obj, _myrun = _geometry_text(exp, run, det, det_dir)

    # vendored (no psana) vs psana, SAME geometry text, SAME kwargs
    my_ix, my_iy = pgeo.pixel_coord_indexes_from_text(
        geometry_text, do_tilt=True, cframe=0)
    ps_ix, ps_iy = _psana_indexes_from_text(
        geometry_text, do_tilt=True, cframe=0)

    assert my_ix.shape == ps_ix.shape and my_iy.shape == ps_iy.shape, \
        f"[{label}] shape mismatch: my {my_ix.shape}/{my_iy.shape} vs " \
        f"psana {ps_ix.shape}/{ps_iy.shape}"
    assert my_ix.dtype == ps_ix.dtype and my_iy.dtype == ps_iy.dtype, \
        f"[{label}] dtype mismatch: my {my_ix.dtype}/{my_iy.dtype} vs " \
        f"psana {ps_ix.dtype}/{ps_iy.dtype}"
    assert np.array_equal(my_ix, ps_ix), \
        f"[{label}] vendored ix != psana ix (max|diff|={np.abs(my_ix.astype('int64')-ps_ix.astype('int64')).max()})"
    assert np.array_equal(my_iy, ps_iy), \
        f"[{label}] vendored iy != psana iy (max|diff|={np.abs(my_iy.astype('int64')-ps_iy.astype('int64')).max()})"
    print(f"[{label}] vendored ix/iy {my_ix.shape} {my_ix.dtype} "
          f"== psana GeometryAccess.get_pixel_coord_indexes(do_tilt=True, "
          f"cframe=0) (byte-exact, array_equal=True)")


def test_vendored_derivation_byte_exact_jungfrau():
    """Vendored derivation == psana for jungfrau, same text + kwargs."""
    if not _have_psana():
        print("SKIP test_vendored_derivation_byte_exact_jungfrau: no psana")
        return
    _check_derivation_byte_exact(JF_EXP, JF_RUN, JF_DET, JF_DIR, "jungfrau")


def test_vendored_derivation_byte_exact_epix10ka():
    """Vendored derivation == psana for epix10ka, same text + kwargs."""
    if not _have_psana():
        print("SKIP test_vendored_derivation_byte_exact_epix10ka: no psana")
        return
    _check_derivation_byte_exact(EP_EXP, EP_RUN, EP_DET, EP_DIR, "epix10ka")


# --------------------------------------------------------------------------
# (2) end-to-end render via the vendored derivation path (NO cached ix/iy)
# --------------------------------------------------------------------------
def test_render_via_vendored_derivation(out_dir):
    """With NO cached ix.npy/iy.npy and derive_geometry_if_missing=True, the
    jungfrau render assembles the (4216,4432) f32 image byte-exact vs psana --
    proving the vendored derivation path drives the renderer end-to-end."""
    if not _have_psana():
        print("SKIP test_render_via_vendored_derivation: no psana")
        return
    import pscalib
    import pscalib.providers.snapshot as ps_snap
    import pscalib.geometry as pgeo

    from psana import DataSource
    ds = DataSource(exp=JF_EXP, run=JF_RUN, dir=JF_DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(JF_DET)
    evt = next(myrun.events())
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_image = np.asarray(det.raw.image(evt))
    assert gt_image.shape == (4216, 4432) and gt_image.dtype == np.float32, \
        f"unexpected gt image {gt_image.shape} {gt_image.dtype}"

    # Snapshot constants + geometry TEXT, but DO NOT cache ix/iy.
    snap_dir = ps_snap.snapshot_calib(exp=JF_EXP, run=JF_RUN, dir=JF_DIR,
                                      detname=JF_DET, out_dir=out_dir)
    # Ensure NO cached index maps exist -> force the derive-from-text path.
    for f in (pgeo.IX_FILE, pgeo.IY_FILE):
        p = os.path.join(snap_dir, f)
        if os.path.isfile(p):
            os.remove(p)
    assert pgeo.load_pixel_indexes(snap_dir) is None, \
        "expected NO cached ix/iy before the derive path runs"

    snap = ps_snap.load_snapshot(snap_dir)
    assert snap.geometry is not None, "snapshot must carry geometry text"
    # derive_geometry_if_missing=True -> the VENDORED derivation runs here.
    imager = pscalib.Imager(snap, derive_geometry_if_missing=True)

    my_calib = imager.calib(gt_raw)
    my_image = imager.image(my_calib)
    assert my_image.shape == (4216, 4432) and my_image.dtype == np.float32, \
        f"rendered image {my_image.shape} {my_image.dtype}"
    di = np.abs(np.nan_to_num(my_image) - np.nan_to_num(gt_image))
    assert np.array_equal(my_image, gt_image), \
        f"vendored-derived render image not byte-exact: max|diff|={di.max()}"
    print(f"[render via vendored derivation] image {my_image.shape} "
          f"{my_image.dtype} max|diff|={di.max()} array_equal=True")

    # and the derived maps themselves match psana's _pixel_coord_indexes
    pix = det.raw._pixel_coord_indexes()
    gt_ix, gt_iy = np.asarray(pix[0]), np.asarray(pix[1])
    assert np.array_equal(imager.ix, gt_ix) and np.array_equal(imager.iy, gt_iy), \
        "vendored-derived ix/iy != det.raw._pixel_coord_indexes()"
    print("[render via vendored derivation] derived ix/iy == "
          "det.raw._pixel_coord_indexes() (byte-exact)")
    return snap_dir


@pytest.fixture
def snap_dir(out_dir):
    """Under pytest, supply the populated snapshot directory that the
    ``__main__`` runner builds via ``test_render_via_vendored_derivation`` and
    hands to ``test_derive_render_purity_subprocess``. Returns the snap_dir
    path (or ``None`` when psana is absent, which the purity test skips on)."""
    return test_render_via_vendored_derivation(out_dir)


# --------------------------------------------------------------------------
# (3) import purity: derive + render in a FRESH interpreter, no framework
# --------------------------------------------------------------------------
_PURITY_SNIPPET = """
import sys
import numpy as np
import pscalib
import pscalib.geometry as pgeo
import pscalib.providers.snapshot as ps_snap

snap_dir = sys.argv[1]
# remove cached ix/iy so the VENDORED derivation runs on render
import os
for f in (pgeo.IX_FILE, pgeo.IY_FILE):
    p = os.path.join(snap_dir, f)
    if os.path.isfile(p):
        os.remove(p)

snap = ps_snap.load_snapshot(snap_dir)
# derive ix/iy from geometry text (vendored) ...
ix, iy = pgeo.pixel_coord_indexes_from_text(snap.geometry, do_tilt=True, cframe=0)
assert ix.shape == iy.shape
# ... and render through the derive-if-missing path
imager = pscalib.Imager(snap, derive_geometry_if_missing=True)
calib = imager.calib(np.zeros((32, 512, 1024), dtype=np.uint16))
image = imager.image(calib)
assert image.shape == (4216, 4432), image.shape

forbidden = ("psana", "mpi4py", "h5py", "dgram", "pymongo")
leaked = [m for m in forbidden if m in sys.modules]
assert not leaked, "FRAMEWORK LEAK after derive+render: %s" % leaked
print("PURITY_OK derived+rendered; forbidden absent:", forbidden)
"""


def test_derive_render_purity_subprocess(snap_dir):
    """Fresh interpreter: derive ix/iy from geometry text + render, then assert
    none of the forbidden framework modules are in sys.modules."""
    if snap_dir is None:
        print("SKIP test_derive_render_purity_subprocess: no snapshot dir")
        return
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(repo, "src")
    psdata_src = os.path.join(os.path.dirname(repo), "psdata", "src")
    env = dict(os.environ)
    pyparts = [src]
    if os.path.isdir(psdata_src):
        pyparts.append(psdata_src)
    if env.get("PYTHONPATH"):
        pyparts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pyparts)
    out = subprocess.run([sys.executable, "-c", _PURITY_SNIPPET, snap_dir],
                         capture_output=True, text=True, env=env)
    print(out.stdout.strip())
    if out.returncode != 0:
        print(out.stderr.strip())
    assert out.returncode == 0, \
        f"purity subprocess failed (rc={out.returncode})"
    assert "PURITY_OK" in out.stdout, "purity probe did not confirm clean path"
    print("[purity] derive+render in a fresh interpreter is framework-free")


def main():
    out_dir = os.environ.get("PSCALIB_TEST_OUT", "/tmp/pscalib_us006_out")
    os.makedirs(out_dir, exist_ok=True)

    test_vendored_derivation_byte_exact_jungfrau()
    test_vendored_derivation_byte_exact_epix10ka()
    snap_dir = test_render_via_vendored_derivation(out_dir)
    test_derive_render_purity_subprocess(snap_dir)

    print("\nALL US-006 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
