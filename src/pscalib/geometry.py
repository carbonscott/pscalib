"""pscalib.geometry -- geometry text -> per-pixel image index maps.

The pixel-coordinate index maps ``ix, iy`` (mapping each ``(seg, row, col)``
data pixel to its ``(image_row, image_col)``) are derived from the detector's
geometry text via the **vendored** pure-numpy ``GeometryAccess``
(:mod:`pscalib._geometry`) --
``GeometryAccess.load_pars_from_str/file`` +
``get_pixel_coord_indexes(do_tilt=True, cframe=0)``.  Verified byte-identical
(``np.array_equal``) to ``det.raw._pixel_coord_indexes()``.

This derivation is a *prep/snapshot-time* step, not an apply-time one: the
resulting ``ix.npy`` / ``iy.npy`` are run-pinned constants (the geometry is
fixed for a run), cached once alongside a calibration snapshot.  At render time
:mod:`pscalib.image` consumes the cached ``ix``/``iy`` with numpy only.

``GeometryAccess`` was vendored into :mod:`pscalib._geometry` (US-006) so that
deriving the index maps no longer imports psana at all -- the whole apply/render
path is now framework-free.  The original psana ``GeometryAccess`` was the last
real psana import on that path; the only remaining lazy psana touch in pscalib
is the snapshot *capture* in :mod:`pscalib.providers.snapshot` (superseded by
the webdb provider).
"""

import hashlib
import os

import numpy as np

#: Filenames the index maps are cached under inside a snapshot directory.
IX_FILE = "pixel_index_ix.npy"
IY_FILE = "pixel_index_iy.npy"

#: Filename of the geometry file the index maps are derived from (written by
#: :func:`pscalib.providers.snapshot.snapshot_calib`).
GEOMETRY_FILE = "geometry.txt"

#: Sidecar recording the SHA-256 content fingerprint of the geometry document
#: the cached ``ix``/``iy`` were derived from -- the piece of the cache KEY that
#: was missing (CAL-09).  The index maps live under a snapshot dir keyed only by
#: ``{detname}_r{run:04d}`` (detector + run); nothing recorded *which geometry*
#: produced them, so a detector move (a new geometry doc dropped into the same
#: dir) left the stale ``ix``/``iy`` in place and the image was silently
#: assembled at the OLD pixel positions.  Keying the cache on this fingerprint
#: makes a changed geometry a cache MISS (re-derive) while an unchanged geometry
#: stays a HIT (the performance win is preserved).  Absent on legacy snapshots
#: (written before this fix); treated as *unverifiable* rather than stale so
#: those still load.
GEOM_FP_FILE = "pixel_index_geom.sha256"


def _geometry_fingerprint(geometry_bytes):
    """SHA-256 hex digest of the raw geometry-document bytes.

    This is the content hash that keys the cached ``ix``/``iy`` to the exact
    geometry text that produced them, so two DIFFERENT geometries for the same
    detector get DIFFERENT cache keys (CAL-09).  Hashing the raw bytes (not a
    decoded-then-re-encoded string) keeps the fingerprint independent of any
    text round-trip.
    """
    return hashlib.sha256(geometry_bytes).hexdigest()


def _read_geometry_bytes(geo_path):
    """Read a geometry file as raw bytes (for both derivation and fingerprint)."""
    with open(geo_path, "rb") as fh:
        return fh.read()


def _read_cached_fingerprint(snap_dir):
    """Return the fingerprint recorded alongside the cached maps, or ``None``
    when no sidecar exists (a legacy, pre-fix snapshot)."""
    fp_path = os.path.join(snap_dir, GEOM_FP_FILE)
    if not os.path.isfile(fp_path):
        return None
    with open(fp_path, encoding="utf-8") as fh:
        return fh.read().strip()


def _write_cached_fingerprint(snap_dir, fingerprint):
    """Record the geometry fingerprint the freshly-derived maps came from."""
    fp_path = os.path.join(snap_dir, GEOM_FP_FILE)
    with open(fp_path, "w", encoding="utf-8") as fh:
        fh.write(fingerprint + "\n")


def pixel_coord_indexes_from_text(geometry_text, do_tilt=True, cframe=0):
    """Derive ``(ix, iy)`` per-pixel image index maps from geometry text.

    Uses the vendored pure-numpy ``GeometryAccess``
    (:mod:`pscalib._geometry`, no psana import).  The result is byte-identical
    to ``det.raw._pixel_coord_indexes()`` (== psana's
    ``GeometryAccess.get_pixel_coord_indexes(do_tilt=True, cframe=0)``).

    Parameters
    ----------
    geometry_text : str
        The geometry definition text (e.g.
        ``CalibSnapshot.geometry`` / ``det.raw._calibconst['geometry'][0]``).
    do_tilt : bool
        Apply per-segment tilt (default True -- the ``det.raw.image`` default).
    cframe : int
        Coordinate frame (default 0 -- psana's default).

    Returns
    -------
    (ix, iy) : (ndarray, ndarray)
        Per-pixel image row / column index maps, shaped as the data
        (``(nsegs, 512, 1024)`` for Jungfrau), dtype as psana returns
        (``uint64``).
    """
    # Vendored numpy-only GeometryAccess (US-006) -- no psana import.  Imported
    # at call time only to keep ``import pscalib.geometry`` itself trivially
    # cheap; the chain is pure os/numpy/math/logging.
    from ._geometry.GeometryAccess import GeometryAccess

    geo = GeometryAccess()
    geo.load_pars_from_str(geometry_text)
    ix, iy = geo.get_pixel_coord_indexes(do_tilt=do_tilt, cframe=cframe)
    return np.asarray(ix), np.asarray(iy)


def cache_pixel_indexes_for_snapshot(snap_dir, do_tilt=True, cframe=0,
                                     overwrite=False):
    """Derive and cache ``ix.npy``/``iy.npy`` into a calib snapshot dir.

    Reads ``geometry.txt`` from the snapshot (written by
    :func:`pscalib.providers.snapshot.snapshot_calib`), derives the index maps
    with the vendored numpy-only ``GeometryAccess`` (no psana), and writes
    :data:`IX_FILE` / :data:`IY_FILE` (plus a :data:`GEOM_FP_FILE` fingerprint)
    next to the constants.  This is the one-time augmentation that makes the
    snapshot self-sufficient for a fully-offline render.

    Cache key (CAL-09)
    ------------------
    The cached maps are only reused (the ``overwrite=False`` fast path) when the
    recorded geometry fingerprint matches the *current* ``geometry.txt``.  A
    changed geometry -- a detector move dropping a new geometry doc into the
    same ``{detname}_r{run:04d}`` dir -- no longer matches, so the stale maps are
    a cache MISS and get re-derived instead of silently reused.  An unchanged
    geometry still matches, so the derivation is skipped (the performance win).

    Parameters
    ----------
    snap_dir : str
        A snapshot directory (``{detname}_r{run:04d}/``) containing
        ``geometry.txt``.
    do_tilt, cframe :
        Passed to :func:`pixel_coord_indexes_from_text`.
    overwrite : bool
        If True, always re-derive.  If False (default), reuse the cached maps
        *only* when their geometry fingerprint still matches ``geometry.txt``.

    Returns
    -------
    (ix_path, iy_path) : (str, str)
        Absolute paths of the written (or reused) index files.
    """
    ix_path = os.path.join(snap_dir, IX_FILE)
    iy_path = os.path.join(snap_dir, IY_FILE)
    geo_path = os.path.join(snap_dir, GEOMETRY_FILE)

    have_maps = os.path.isfile(ix_path) and os.path.isfile(iy_path)
    geom_bytes = _read_geometry_bytes(geo_path) if os.path.isfile(geo_path) else None
    want_fp = _geometry_fingerprint(geom_bytes) if geom_bytes is not None else None

    if not overwrite and have_maps:
        if geom_bytes is None:
            # No geometry to validate against and none to re-derive from --
            # return the cached maps as-is (unchanged legacy behavior).
            return os.path.abspath(ix_path), os.path.abspath(iy_path)
        if _read_cached_fingerprint(snap_dir) == want_fp:
            # cache HIT: the maps were derived from this very geometry.
            return os.path.abspath(ix_path), os.path.abspath(iy_path)
        # else: fingerprint absent (legacy) or MISMATCHED (the geometry doc
        # changed -- a detector move) -> STALE -> fall through and re-derive.

    if geom_bytes is None:
        raise FileNotFoundError(
            f"no {GEOMETRY_FILE} in {snap_dir!r} -- snapshot has no geometry to "
            f"derive pixel indexes from")
    geometry_text = geom_bytes.decode("utf-8")

    ix, iy = pixel_coord_indexes_from_text(geometry_text, do_tilt=do_tilt,
                                           cframe=cframe)
    np.save(ix_path, ix, allow_pickle=False)
    np.save(iy_path, iy, allow_pickle=False)
    _write_cached_fingerprint(snap_dir, want_fp)
    return os.path.abspath(ix_path), os.path.abspath(iy_path)


def load_pixel_indexes(snap_dir):
    """Load cached ``(ix, iy)`` index maps from a snapshot dir (pure numpy).

    Returns ``None`` when the maps are not usable, so the caller re-derives:

    * they have not been cached yet (call
      :func:`cache_pixel_indexes_for_snapshot` once to create them), or
    * they are STALE with respect to the snapshot's current geometry (CAL-09) --
      the snapshot carries both ``geometry.txt`` and a :data:`GEOM_FP_FILE`
      fingerprint sidecar, and the sidecar does NOT match the current geometry
      (a detector move rewrote ``geometry.txt`` under the same snapshot dir).

    When staleness cannot be verified -- a legacy snapshot with no fingerprint
    sidecar, or one that does not keep its geometry text -- the maps are loaded
    as before (backward compatible).  ``render.py`` calls this *first*, so
    refusing a stale map here is what makes the render path re-derive after a
    detector move rather than silently assembling at the old pixel positions.
    """
    ix_path = os.path.join(snap_dir, IX_FILE)
    iy_path = os.path.join(snap_dir, IY_FILE)
    if not (os.path.isfile(ix_path) and os.path.isfile(iy_path)):
        return None

    geo_path = os.path.join(snap_dir, GEOMETRY_FILE)
    cached_fp = _read_cached_fingerprint(snap_dir)
    if cached_fp is not None and os.path.isfile(geo_path):
        want_fp = _geometry_fingerprint(_read_geometry_bytes(geo_path))
        if cached_fp != want_fp:
            # cached maps were derived from a DIFFERENT geometry -> stale miss.
            return None

    return (np.load(ix_path, allow_pickle=False),
            np.load(iy_path, allow_pickle=False))
