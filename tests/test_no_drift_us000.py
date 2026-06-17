#!/usr/bin/env python3
"""US-000 (d): no duplicate-canonical hazard between pscalib and psdata.

US-000 acceptance criterion (d): there must NOT be two independently-editable
copies of the calibration/image engine.

Two options were on the table to satisfy (d):
  (a) psdata's ``calib`` / ``hdr`` modules become a RE-EXPORT SHIM of pscalib;
  (b) psdata RETIRES its ``calib`` / ``hdr`` modules entirely and pscalib becomes
      the SOLE canonical home (psdata stays a pure framework-free reader).

The design that SHIPPED is option (b): psdata's commit "refactor: retire
psdata.calib/hdr shims -- psdata is reader-only" removed those modules. So this
test no longer asserts a shim by object identity; it asserts the *stronger*
guarantee that (b) gives for free -- there is exactly ONE copy of the engine
(in pscalib), because psdata no longer ships one at all:

  1. ``import psdata.calib`` / ``import psdata.hdr`` raise ModuleNotFoundError
     (the shims are gone -- no duplicate-canonical hazard, structurally).
  2. ``psdata`` itself still imports (pscalib depends on it as the reader) but
     does NOT expose the calibration engine symbols.
  3. The canonical engine lives in pscalib and imports fine.

It is pure-numpy and needs no psana (it never opens a DataSource).
"""

import os
import sys

import numpy as np  # noqa: F401  (import-purity backdrop)

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")          # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
# psdata is a sibling standalone package; add its src too so the test is robust
# whether or not PYTHONPATH already carries it (mirrors run_tests.sh).
_PSDATA_SRC = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "psdata", "src")
if os.path.isdir(_PSDATA_SRC) and _PSDATA_SRC not in sys.path:
    sys.path.insert(0, _PSDATA_SRC)


def test_psdata_calib_hdr_shims_are_retired():
    """psdata.calib / psdata.hdr no longer exist -- shipped design is option (b)."""
    import psdata  # the reader dependency; must import
    for mod in ("psdata.calib", "psdata.hdr"):
        with pytest.raises(ModuleNotFoundError):
            __import__(mod)
    # psdata's top level must NOT re-expose the calibration engine either.
    for leaked in ("snapshot_calib", "calib_jungfrau", "assemble_image",
                   "HDRImager", "Imager"):
        assert not hasattr(psdata, leaked), (
            f"psdata re-exposes {leaked!r} -- the engine leaked back into the "
            f"reader; pscalib must be the sole canonical home")
    print("[no-drift] psdata.calib/psdata.hdr are retired; psdata exposes no "
          "calibration engine -- pscalib is the sole canonical home")


def test_pscalib_is_the_canonical_engine():
    """The one canonical copy of the engine lives in pscalib and imports."""
    import pscalib
    import pscalib.providers.snapshot as ps_snap

    # snapshot provider symbols
    for name in ("snapshot_calib", "load_snapshot", "CalibSnapshot"):
        assert hasattr(ps_snap, name), f"pscalib.providers.snapshot.{name} missing"
        assert getattr(pscalib, name, None) is getattr(ps_snap, name), (
            f"pscalib.{name} is not re-exported from providers.snapshot")
    # apply + image/geometry engine symbols on the pscalib top level
    for name in ("calib_jungfrau", "calib_epix10ka", "assemble_image", "Imager",
                 "pixel_coord_indexes_from_text", "cache_pixel_indexes_for_snapshot",
                 "load_pixel_indexes"):
        assert hasattr(pscalib, name), f"pscalib.{name} missing -- canonical engine incomplete"
    print("[no-drift] pscalib holds the one canonical calib/image engine")


def test_canonical_engine_import_is_framework_free():
    """Importing pscalib's engine must stay numpy-only (no framework leak).

    This is an IN-PROCESS check: it is only meaningful in a clean interpreter.
    Under a full pytest run a sibling cross-check test may have already imported
    psana into this process; in that case skip and defer to the authoritative
    fresh-subprocess gate (``test_purity_us007.py``). In ``__main__`` mode the
    interpreter is clean, so it asserts for real.
    """
    import pscalib
    already = [m for m in pscalib.FORBIDDEN_MODULES
               if any(n == m or n.startswith(m + ".") for n in sys.modules)]
    if already:
        msg = (f"in-proc purity skipped: sibling already imported {already}; "
               f"the fresh-interpreter subprocess check (test_purity_us007) is "
               f"authoritative")
        print("[no-drift] " + msg)
        pytest.skip(msg)
    pscalib.assert_no_framework_imports()
    print("[no-drift] importing the pscalib engine stays framework-free")


def main():
    print("=" * 72)
    print("US-000 (d) no-drift: psdata is reader-only; pscalib is the sole "
          "canonical engine")
    print("=" * 72)
    test_psdata_calib_hdr_shims_are_retired()
    test_pscalib_is_the_canonical_engine()
    test_canonical_engine_import_is_framework_free()
    print("\nALL US-000 (d) NO-DRIFT CHECKS PASSED")


if __name__ == "__main__":
    main()
