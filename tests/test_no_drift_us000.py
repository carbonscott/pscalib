#!/usr/bin/env python3
"""US-000 (d): no duplicate-canonical hazard between pscalib and psdata.

US-000 acceptance criterion (d): psdata's ``calib/`` + ``hdr/`` are RETAINED
(psdata is under active development -- not deleted), but there must NOT be two
independently-editable copies of the calibration engine.

The chosen option (recorded in the story-completion notes) is OPTION (a):
psdata's ``calib`` / ``hdr`` modules are a RE-EXPORT SHIM of pscalib.  pscalib
holds the one canonical implementation; psdata re-exports it.  Drift is then
*structurally impossible* -- there is only one copy of each function/class
object, and this test proves it by identity (``is``), not by text comparison.

This test imports BOTH packages and asserts the shared public symbols are the
SAME objects.  It is pure-numpy and needs no psana (it never opens a
DataSource).  It does require psdata to be importable AND for psdata's shim to
resolve pscalib (pscalib must be on the path / installed alongside psdata).
"""

import os
import sys

import numpy as np  # noqa: F401  (import-purity backdrop)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


def test_psdata_calib_is_pscalib_reexport():
    """psdata.calib's snapshot symbols ARE pscalib's (same objects)."""
    import pscalib.providers.snapshot as ps_snap
    import psdata.calib as pd_calib

    for name in ("snapshot_calib", "load_snapshot", "CalibSnapshot"):
        assert getattr(pd_calib, name) is getattr(ps_snap, name), (
            f"psdata.calib.{name} is NOT the same object as "
            f"pscalib.providers.snapshot.{name} -- the shim drifted")
    print("[no-drift] psdata.calib.{snapshot_calib,load_snapshot,"
          "CalibSnapshot} are pscalib objects (identity)")


def test_psdata_hdr_is_pscalib_reexport():
    """psdata.hdr's render/apply/image symbols ARE pscalib's (same objects)."""
    import pscalib
    import psdata.hdr as pd_hdr

    pairs = [
        ("HDRImager", pscalib.HDRImager),
        ("calib_jungfrau", pscalib.calib_jungfrau),
        ("assemble_image", pscalib.assemble_image),
        ("pixel_coord_indexes_from_text", pscalib.pixel_coord_indexes_from_text),
        ("cache_pixel_indexes_for_snapshot",
         pscalib.cache_pixel_indexes_for_snapshot),
        ("load_pixel_indexes", pscalib.load_pixel_indexes),
    ]
    for name, canonical in pairs:
        assert getattr(pd_hdr, name) is canonical, (
            f"psdata.hdr.{name} is NOT the same object as pscalib.{name} "
            f"-- the shim drifted")
    print("[no-drift] psdata.hdr.{HDRImager,calib_jungfrau,assemble_image,"
          "geometry fns} are pscalib objects (identity)")


def test_shim_import_purity():
    """Importing the psdata shim (which re-exports pscalib) must stay numpy-only
    -- no framework leaks via the re-export."""
    import pscalib
    import psdata.calib  # noqa: F401
    import psdata.hdr    # noqa: F401
    pscalib.assert_no_framework_imports()
    for m in pscalib.FORBIDDEN_MODULES:
        assert m not in sys.modules, f"{m} leaked via the psdata shim import"
    print("[no-drift] importing the psdata re-export shim stays framework-free")


def main():
    print("=" * 72)
    print("US-000 (d) no-drift: psdata calib/hdr are a pscalib re-export shim")
    print("=" * 72)
    test_psdata_calib_is_pscalib_reexport()
    test_psdata_hdr_is_pscalib_reexport()
    test_shim_import_purity()
    print("\nALL US-000 (d) NO-DRIFT CHECKS PASSED")


if __name__ == "__main__":
    main()
