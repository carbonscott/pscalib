#!/usr/bin/env python3
"""CAL-09 regression: the geometry -> ix/iy cache must be keyed by the geometry.

Bug (CAL-09): ``pscalib`` derives the per-pixel image-index maps ``ix``/``iy``
from the detector geometry and MEMOIZES them on disk inside a snapshot dir keyed
only by ``{detname}_r{run:04d}`` (detector + run) plus the mere existence of the
``.npy`` files.  Nothing in that key records *which geometry document* produced
the maps.  Geometry is run-dependent: after a detector is physically moved a new
geometry doc applies, but the SAME cache key returns the STALE ``ix``/``iy`` from
the previous geometry -- the image is silently assembled at the OLD pixel
positions, with no error.

This probe is the pre-fix/post-fix discriminator.  It is FULLY SELF-CONTAINED --
numpy only, NO psana, NO SLAC data -- so it runs anywhere, and it contains NO
part of the fix (it only touches the public geometry API and asserts on outcomes;
it never mentions the fingerprint sidecar the fix adds).

It builds two DIFFERENT geometries for the SAME detector (a real "detector move":
one JUNGFRAU:V2 segment mounted at 0 deg vs the same segment rotated 90 deg in
its mount -- a rotation provably relabels every pixel's image row/column, unlike
a pure translation, which the auto-cropped image frame leaves invariant), caches
the maps for geometry A, then swaps in geometry B under the SAME snapshot dir and
re-runs the cache.

  * On the PARENT (cache keyed without geometry identity): the second call sees
    the ``.npy`` files already exist and returns geometry A's STALE maps ->
    the "maps reflect B" assertion fails -> exit 1.
  * On the FIX (cache keyed by the geometry document): the changed geometry is a
    cache MISS, the maps are re-derived from B, and the assertion passes.

It also asserts that the SAME geometry still HITS the cache (via a call counter
on the expensive derivation) -- proving the fix keys the cache correctly rather
than simply disabling it.
"""

import os
import shutil
import sys
import tempfile

import numpy as np

# --- resolve pscalib from this repo's src/, robust to cwd (no PYTHONPATH needed).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pscalib.geometry as pgeo  # noqa: E402

#: The geometry file a snapshot dir carries; the cache derives ix/iy from it.
#: (Literal, not a pscalib symbol, so this probe imports identically on the
#: pre-fix parent -- which hard-codes this name -- and on the fixed tree.)
GEOMETRY_FILE = "geometry.txt"


def _geometry_text(rot_z_deg):
    """A minimal single-JUNGFRAU:V2-segment geometry, rotated ``rot_z_deg`` in
    plane about its mount.  Columns are psana's geometry-file format:
    ``pname pindex oname oindex x0 y0 z0 rot_z rot_y rot_x tilt_z tilt_y tilt_x``.
    The parent frame ``DET:V1`` is auto-created by GeometryAccess as the top
    object; the ``JUNGFRAU:V2`` leaf carries the pixel coordinates.
    """
    return (
        "DET:V1  0  JUNGFRAU:V2  0   "
        f"0.0 0.0 0.0   {rot_z_deg} 0 0   0 0 0\n"
    )


def _load_cached_maps(snap_dir):
    """Read whatever ix/iy the cache currently has on disk (bypassing any
    load-time validation, so we observe exactly what the cache call left)."""
    ix = np.load(os.path.join(snap_dir, pgeo.IX_FILE), allow_pickle=False)
    iy = np.load(os.path.join(snap_dir, pgeo.IY_FILE), allow_pickle=False)
    return ix, iy


def _run_probe():
    geom_a = _geometry_text(0)     # detector mounted at 0 deg
    geom_b = _geometry_text(90)    # SAME detector, physically rotated 90 deg

    # Ground truth derived directly from each geometry (real function, before we
    # install the call counter).  These are what a CORRECT cache must return.
    ix_a, iy_a = pgeo.pixel_coord_indexes_from_text(geom_a)
    ix_b, iy_b = pgeo.pixel_coord_indexes_from_text(geom_b)

    # Premise: the two geometries must genuinely move pixels, else the probe is
    # vacuous.  (A 90 deg rotation flips the image frame, relabelling ix and iy.)
    assert not (np.array_equal(ix_a, ix_b) and np.array_equal(iy_a, iy_b)), (
        "test premise broken: geometry A and B produced identical ix/iy -- pick "
        "a geometry change that actually moves pixels")

    # Count how often the expensive geometry->index derivation actually runs, so
    # we can prove (a) a changed geometry re-derives and (b) an unchanged one
    # does not.  Patch the module global the cache calls through.
    orig_derive = pgeo.pixel_coord_indexes_from_text
    calls = {"n": 0}

    def _counting_derive(*args, **kwargs):
        calls["n"] += 1
        return orig_derive(*args, **kwargs)

    pgeo.pixel_coord_indexes_from_text = _counting_derive

    snap_dir = tempfile.mkdtemp(prefix="pscalib_cal09_")
    try:
        geo_path = os.path.join(snap_dir, GEOMETRY_FILE)

        # --- (1) cache the maps for geometry A -------------------------------
        with open(geo_path, "w", encoding="utf-8") as fh:
            fh.write(geom_a)
        pgeo.cache_pixel_indexes_for_snapshot(snap_dir)
        assert calls["n"] == 1, (
            "expected the first cache call to derive the maps once "
            f"(derivations={calls['n']})")
        ix_disk, iy_disk = _load_cached_maps(snap_dir)
        assert np.array_equal(ix_disk, ix_a) and np.array_equal(iy_disk, iy_a), (
            "the first cache call did not store geometry A's maps")

        # --- (2) the detector moves: swap geometry B into the SAME dir -------
        with open(geo_path, "w", encoding="utf-8") as fh:
            fh.write(geom_b)
        pgeo.cache_pixel_indexes_for_snapshot(snap_dir)

        # THE DISCRIMINATOR: after a geometry change the cache must hold B's
        # maps, not the stale A maps.  On the parent the second call sees the
        # .npy files exist and returns A -> this fails.
        ix_disk, iy_disk = _load_cached_maps(snap_dir)
        assert np.array_equal(ix_disk, ix_b) and np.array_equal(iy_disk, iy_b), (
            "STALE CACHE (CAL-09): after the geometry changed from A to B the "
            "cache still holds geometry A's ix/iy -- the cache key does not "
            "include the geometry document identity")
        assert not (np.array_equal(ix_disk, ix_a)
                    and np.array_equal(iy_disk, iy_a)), (
            "STALE CACHE (CAL-09): the cached maps still equal geometry A's")

        # The render path loads via load_pixel_indexes FIRST; it too must not
        # hand back the stale maps after a detector move.
        loaded = pgeo.load_pixel_indexes(snap_dir)
        assert loaded is not None, "cached maps unexpectedly missing"
        assert (np.array_equal(loaded[0], ix_b)
                and np.array_equal(loaded[1], iy_b)), (
            "STALE CACHE (CAL-09): load_pixel_indexes returned geometry A's "
            "stale maps for a snapshot whose geometry is now B")

        # --- (3) the same geometry must STILL hit the cache ------------------
        # (proves the fix keys the cache by geometry, not that it disabled it.)
        n_before = calls["n"]
        pgeo.cache_pixel_indexes_for_snapshot(snap_dir)   # geometry unchanged (B)
        assert calls["n"] == n_before, (
            "the SAME geometry must be a cache HIT (no re-derivation); the cache "
            f"re-derived unnecessarily (derivations went {n_before} -> "
            f"{calls['n']}) -- the performance win was lost")

        print("[CAL-09] geometry-keyed cache OK: changed geometry re-derived "
              f"(total derivations={calls['n']}), unchanged geometry hit the "
              "cache, and no stale ix/iy were served.")
    finally:
        pgeo.pixel_coord_indexes_from_text = orig_derive
        shutil.rmtree(snap_dir, ignore_errors=True)


def test_cal09_geometry_cache_keyed_by_geometry_identity():
    """The ix/iy cache must invalidate when the geometry document changes."""
    _run_probe()


def main():
    test_cal09_geometry_cache_keyed_by_geometry_identity()
    print("\nCAL-09 REGRESSION PROBE PASSED")


if __name__ == "__main__":
    main()
