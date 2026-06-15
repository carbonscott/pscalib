#!/usr/bin/env python3
"""US-001 acceptance test: vendored requests-only web-DB retrieval provider.

Verifies the US-001 acceptance criteria for the reference Jungfrau dataset
(exp=mfx100848724, run=51, dir=/sdf/data/lcls/ds/prj/public01/xtc, det=jungfrau):

  (a) BYTE-EXACT RETRIEVAL -- in a process that NEVER opens a DataSource,
      ``webdb.get_constants(uniqueid, exp='mfx100848724', run=51)`` -- where
      ``uniqueid`` is ``det.raw._uniqueid`` -- returns pedestals, pixel_gain,
      pixel_offset, pixel_status, pixel_rms arrays sha1-identical to psana's
      ``det.raw._calibconst[ctype][0]`` for that run.  (psana is used here ONLY
      to read the ground truth + the uniqueid; the webdb fetch opens no
      DataSource and imports no psana.)

  (b) IMPORT PURITY -- in a FRESH interpreter that never imports psana: after a
      live web fetch, ``sys.modules`` contains 'requests' but NOT any of
      ('psana','mpi4py','dgram','pymongo').  Importing pscalib stays numpy-only
      (no 'requests' until the opt-in webdb provider is imported).

This test needs the PRODUCTION psana env (psconda.sh) to GENERATE the ground
truth -- run it on sdfiana025 via ``run_tests.sh tests/test_webdb_us001.py``
(it puts pscalib/src + psdata/src on PYTHONPATH).  Both gates need network
access to the on-site calib web service (psdmint).  The byte-exact gate skips
cleanly (with a message) if psana is not importable; the purity gate runs
regardless (it needs only the web deps + network).
"""

import hashlib
import os
import subprocess
import sys

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

# The five ctypes the US-001 byte-exact gate names explicitly.
GATE_CTYPES = ("pedestals", "pixel_gain", "pixel_offset",
               "pixel_status", "pixel_rms")

# After a live web fetch, requests is expected; none of these may appear.
FORBIDDEN_WEB = ("psana", "mpi4py", "dgram", "pymongo")
# The pscalib whole-import forbidden set (extends psdata's).
FORBIDDEN_ALL = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


def _sha1_arr(x):
    """sha1 of an ndarray's bytes (C-contiguous) or a str's utf-8 bytes."""
    if isinstance(x, np.ndarray):
        return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()
    if isinstance(x, str):
        return hashlib.sha1(x.encode()).hexdigest()
    raise TypeError("unexpected constant payload type: %s" % type(x).__name__)


# --------------------------------------------------------------------------
# (a) byte-exact retrieval vs psana _calibconst, with NO DataSource in webdb
# --------------------------------------------------------------------------
def test_webdb_byte_exact():
    """webdb.get_constants(uniqueid, exp, run) returns the named ctypes
    sha1-identical to psana's det.raw._calibconst[ctype][0]."""
    # psana ground truth + the uniqueid (the only psana use here).
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    gt_cc = det.raw._calibconst
    uniqueid = det.raw._uniqueid
    assert gt_cc is not None, "psana _calibconst is None (DB unreachable?)"
    for ct in GATE_CTYPES:
        assert ct in gt_cc, f"psana did not return {ct!r}"

    # The vendored web fetch -- opens NO DataSource, imports NO psana.
    from pscalib.providers import webdb
    got = webdb.get_constants(uniqueid, exp=EXP, run=RUN)
    assert got is not None, "webdb.get_constants returned None"
    for ct in GATE_CTYPES:
        assert ct in got, f"webdb did not return {ct!r}"

    # Byte-exact: sha1 of webdb's array == sha1 of psana's array, per ctype.
    for ct in GATE_CTYPES:
        gt_arr = np.asarray(gt_cc[ct][0])
        wb_arr = np.asarray(got[ct][0])
        assert wb_arr.shape == gt_arr.shape and wb_arr.dtype == gt_arr.dtype, (
            f"{ct}: shape/dtype mismatch web={wb_arr.shape}/{wb_arr.dtype} "
            f"psana={gt_arr.shape}/{gt_arr.dtype}")
        assert _sha1_arr(wb_arr) == _sha1_arr(gt_arr), (
            f"{ct}: sha1 mismatch -- web fetch is NOT byte-exact vs psana")
        assert np.array_equal(wb_arr, gt_arr), f"{ct}: arrays differ"
        print(f"[byte-exact] {ct:16s} {wb_arr.shape} {wb_arr.dtype} "
              f"sha1={_sha1_arr(wb_arr)[:12]} == psana (IDENTICAL)")

    # The full dict matches too (every ctype psana returned, not just the five).
    for ct, (gt_payload, _meta) in gt_cc.items():
        assert ct in got, f"webdb missing ctype {ct!r} that psana returned"
        assert _sha1_arr(got[ct][0]) == _sha1_arr(gt_payload), (
            f"{ct}: full-dict sha1 mismatch")
    print(f"[byte-exact] full _calibconst dict ({len(gt_cc)} ctypes) "
          f"sha1-identical web vs psana")


# --------------------------------------------------------------------------
# (b) import purity -- fresh interpreter, live fetch, no framework
# --------------------------------------------------------------------------
def _uniqueid_via_psana():
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    return det.raw._uniqueid


def test_webdb_import_purity_subprocess(uniqueid):
    """In a FRESH interpreter that never imports psana: import pscalib (must be
    numpy-only), import the webdb provider (pulls requests+bson), do a LIVE
    fetch, and assert 'requests' is present but
    ('psana','mpi4py','dgram','pymongo') are absent."""
    code = (
        "import sys\n"
        "import pscalib\n"
        "assert 'requests' not in sys.modules, 'import pscalib pulled requests'\n"
        "pscalib.assert_no_framework_imports()\n"
        f"assert pscalib.FORBIDDEN_MODULES == {FORBIDDEN_ALL!r}, "
        "pscalib.FORBIDDEN_MODULES\n"
        "from pscalib.providers import webdb\n"
        "assert 'requests' in sys.modules, 'webdb did not import requests'\n"
        "assert 'bson' in sys.modules, 'webdb did not import bson'\n"
        f"got = webdb.get_constants({uniqueid!r}, exp={EXP!r}, run={RUN})\n"
        "assert got is not None and 'pedestals' in got, "
        "('live fetch returned: %s' % (list(got) if got else got))\n"
        "assert 'requests' in sys.modules\n"
        f"bad = [m for m in {FORBIDDEN_WEB!r} if m in sys.modules]\n"
        "assert not bad, ('forbidden modules leaked after live fetch: %s' % bad)\n"
        "print('CLEAN', len(got))\n"
    )
    # Preserve the inherited PYTHONPATH (run_tests.sh added pscalib+psdata src).
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (_PKG_PARENT + (os.pathsep + inherited
                                        if inherited else ""))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, (
        "purity subprocess failed:\nSTDOUT:%s\nSTDERR:%s" %
        (out.stdout, out.stderr))
    assert "CLEAN" in out.stdout, out.stdout
    print("[purity] fresh interp: import pscalib numpy-only; webdb pulls "
          "requests+bson; live fetch clean of psana/mpi4py/dgram/pymongo -- "
          + out.stdout.strip())


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-001 acceptance: vendored requests-only web-DB retrieval provider")
    print("=" * 72)

    if not _have_psana():
        print("\n[skip] psana not importable -- byte-exact + purity gates need "
              "psana to read the ground truth uniqueid.  Source psconda.sh on "
              "sdfiana025 and use run_tests.sh.")
        return

    # (a) byte-exact retrieval vs psana _calibconst (webdb opens no DataSource).
    test_webdb_byte_exact()
    print("[ok] (a) webdb byte-exact vs psana _calibconst for all gate ctypes")

    # (b) import purity: fresh interpreter, live fetch, no framework.
    uniqueid = _uniqueid_via_psana()
    test_webdb_import_purity_subprocess(uniqueid)
    print("[ok] (b) web fetch import purity (subprocess, live fetch)")

    print("\nALL US-001 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
