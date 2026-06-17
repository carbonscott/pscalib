"""pscalib.render -- standalone offline calibrated 2-D image render.

Ties together a per-detector gain decode (:mod:`pscalib.apply`), the vendored
image remap (:mod:`pscalib.image`), and a calibration snapshot
(:mod:`pscalib.providers.snapshot`) into a single offline pipeline:

    raw (N,512,1024) uint16  -- e.g. from psdata.run.Event.stack('jungfrau')
        |  gain decode (cached constants)
        v
    calib (N,512,1024) f32   == det.raw.calib(evt)   (max|diff| == 0)
        |  geometry remap (cached pixel index maps)
        v
    image (4216,4432) f32    == det.raw.image(evt)   (max|diff| == 0)

At *render time* this pulls in only numpy -- no psana, no DB, no MPI.  The one
psana-touching step is the one-time snapshot + index-map prep
(:func:`pscalib.providers.snapshot.snapshot_calib` + :mod:`pscalib.geometry`).

The render is per-detector-type (the gain decode and geometry differ by
detector).  Today only Jungfrau is wired in; :class:`Imager` dispatches on
``snapshot.detname`` so other detector types can be added without changing the
public surface.  (US-004/US-005 generalise this to a detector registry; US-000
preserves the lifted jungfrau-only dispatch verbatim.)
"""

import numpy as np

from . import image as _image
from . import geometry as _geometry
from .apply import jungfrau as _jungfrau

#: Detectors with a wired-in gain decode.  Maps the snapshot detname (matched
#: case-insensitively as a prefix) to its calibrate function.
_GAIN_DECODERS = {
    "jungfrau": _jungfrau.calib_jungfrau,
}


def _decoder_for(detname):
    key = (detname or "").lower()
    for name, fn in _GAIN_DECODERS.items():
        if key.startswith(name):
            return fn
    raise NotImplementedError(
        f"no gain decode wired in for detector {detname!r}; "
        f"supported: {sorted(_GAIN_DECODERS)}")


class Imager:
    """Offline calibrated 2-D image renderer pinned to one calib snapshot.

    Construct from a :class:`pscalib.providers.snapshot.CalibSnapshot`.  All
    inputs -- constants, mask, geometry index maps -- come from the snapshot, so
    once built the renderer touches no psana, DB, or network.

    The geometry index maps (``ix``/``iy``) are read from the snapshot if they
    were cached (:func:`pscalib.geometry.cache_pixel_indexes_for_snapshot`),
    else derived once from the snapshot's geometry text via the **vendored**
    numpy-only ``GeometryAccess`` (:mod:`pscalib._geometry`, no psana, US-006)
    and cached for next time.

    Parameters
    ----------
    snapshot : pscalib.providers.snapshot.CalibSnapshot
        A pinned calibration snapshot for the detector + run to render.
    derive_geometry_if_missing : bool
        If the snapshot has no cached ``ix``/``iy`` but does carry geometry
        text, derive them (one vendored numpy-only ``GeometryAccess`` call, no
        psana) and cache them into the snapshot dir.  Default True.  Set False
        to force a construction that fails fast if the maps were never cached.
    """

    def __init__(self, snapshot, derive_geometry_if_missing=True):
        self.snapshot = snapshot
        self.detname = snapshot.detname
        self.run = snapshot.run
        self._calibrate = _decoder_for(self.detname)

        self.pedestals = snapshot.pedestals
        self.pixel_gain = snapshot.pixel_gain
        self.pixel_offset = snapshot.pixel_offset          # may be None
        self.mask = snapshot.mask
        if self.pedestals is None or self.pixel_gain is None:
            raise ValueError(
                f"snapshot {snapshot!r} is missing pedestals/pixel_gain -- "
                f"cannot calibrate")

        idx = _geometry.load_pixel_indexes(snapshot.path)
        if idx is None:
            if not (derive_geometry_if_missing and snapshot.geometry):
                raise ValueError(
                    f"snapshot {snapshot.path!r} has no cached pixel index "
                    f"maps and {'geometry text is absent' if not snapshot.geometry else 'derive_geometry_if_missing=False'}; "
                    f"run pscalib.geometry.cache_pixel_indexes_for_snapshot "
                    f"once (with psana) to create them")
            _geometry.cache_pixel_indexes_for_snapshot(snapshot.path)
            idx = _geometry.load_pixel_indexes(snapshot.path)
        self.ix, self.iy = idx
        # full-detector image-grid extent, pinned once.
        self._rc_tot_max = [int(np.max(self.ix.ravel())),
                            int(np.max(self.iy.ravel()))]

    # -- staleness enforcement (US-002) -----------------------------------
    def check_run(self, run, allow_stale=False):
        """Enforce that this imager's snapshot constants are valid for ``run``
        (refuse-by-default; US-002).

        Delegates to :meth:`CalibSnapshot.check_validity`:

        * in range -- returns silently (empty offender list);
        * out of range and ``allow_stale=False`` (default) -- raises
          :class:`pscalib.model.StaleConstantsError`;
        * out of range and ``allow_stale=True`` -- logs a warning and returns
          the offenders.

        :meth:`calib` / :meth:`render` call this automatically when given a
        ``run`` argument; call it directly to enforce without rendering.
        """
        return self.snapshot.check_validity(run, allow_stale=allow_stale)

    # -- the two products -------------------------------------------------
    def calib(self, raw, run=None, allow_stale=False):
        """Calibrate a raw stack into ADU (``== det.raw.calib(evt)``).

        Parameters
        ----------
        raw : ndarray ``(N, 512, 1024)`` uint16
        run : int | None
            If given, ENFORCE that the snapshot's constants are valid for this
            run *before* calibrating (US-002, refuse-by-default).  Out-of-range
            raises :class:`pscalib.model.StaleConstantsError` unless
            ``allow_stale=True``.  ``None`` (default) skips the check entirely --
            backward-compatible with the pre-US-002 zero-arg call.
        allow_stale : bool
            Downgrade an out-of-range refusal to a logged warning (only
            consulted when ``run`` is not ``None``).

        Returns
        -------
        ndarray ``(N, 512, 1024)`` float32
        """
        if run is not None:
            self.check_run(run, allow_stale=allow_stale)
        return self._calibrate(raw, self.pedestals, self.pixel_gain,
                               self.pixel_offset, self.mask)

    def image(self, calib_or_raw, is_raw=False, run=None, allow_stale=False):
        """Assemble the calibrated 2-D image (``== det.raw.image(evt)``).

        Parameters
        ----------
        calib_or_raw : ndarray
            Either a calibrated stack (default) or, if ``is_raw=True``, a raw
            stack to calibrate first.
        is_raw : bool
            Treat the input as raw uint16 and calibrate it before assembly.
        run : int | None
            Enforce validity for this run when ``is_raw=True`` (ignored
            otherwise -- a pre-calibrated stack has no raw to re-check).  See
            :meth:`calib`.
        allow_stale : bool
            Downgrade an out-of-range refusal to a logged warning.

        Returns
        -------
        ndarray, 2-D, float32  (e.g. ``(4216, 4432)`` for Jungfrau 8M)
        """
        if is_raw:
            calib = self.calib(calib_or_raw, run=run, allow_stale=allow_stale)
        else:
            calib = calib_or_raw
        return _image.assemble_image(calib, self.ix, self.iy,
                                     rc_tot_max=self._rc_tot_max)

    def render(self, raw, run=None, allow_stale=False):
        """Full pipeline: raw -> (calib, image).  Convenience wrapper.

        Parameters
        ----------
        raw : ndarray ``(N, 512, 1024)`` uint16
        run : int | None
            If given, ENFORCE validity for this run before rendering (US-002).
        allow_stale : bool
            Downgrade an out-of-range refusal to a logged warning.

        Returns
        -------
        (calib, image) : (ndarray (N,512,1024) f32, ndarray 2-D f32)
        """
        calib = self.calib(raw, run=run, allow_stale=allow_stale)
        image = self.image(calib)
        return calib, image

    def __repr__(self):
        return (f"Imager(detname={self.detname!r}, run={self.run}, "
                f"image_shape=({self._rc_tot_max[0] + 1},"
                f"{self._rc_tot_max[1] + 1}))")


def from_snapshot_dir(snap_dir, **kwa):
    """Build an :class:`Imager` from a calib snapshot *directory* path.

    Loads the snapshot with
    :func:`pscalib.providers.snapshot.load_snapshot` (pure numpy) then
    constructs the imager.  Convenience for the common "I have a snapshot on
    disk" case.
    """
    from .providers.snapshot import load_snapshot
    return Imager(load_snapshot(snap_dir), **kwa)


#: Deprecated back-compat alias for :class:`Imager`.  The old name carried the
#: misnomer "HDR" (high dynamic range), which names no distinct operation -- the
#: pipeline is just gain-decode calibration + geometry assembly.  Kept so
#: existing ``from pscalib.render import HDRImager`` callers keep working; prefer
#: :class:`Imager` in new code.
HDRImager = Imager
