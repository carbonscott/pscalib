"""pscalib.apply.jungfrau -- Jungfrau 3-gain calibration (vendored numpy).

The per-detector-type gain-decode leaf of the pure-numpy apply engine.  A
faithful, framework-free numpy re-implementation of psana's
``psana.detector.UtilsJungfrau.calib_jungfrau_single_panel`` (looped over
segments, exactly as ``calib_jungfrau`` does), verified byte-identical
(``np.array_equal``) to ``det.raw.calib(evt)`` for the reference Jungfrau 8M
dataset (exp=mfx100848724, run=51, det='jungfrau').

This is the canonical home of the Jungfrau decode -- it was first proven in
psdata's calibration layer, which now re-exports from here.

The Jungfrau is an *auto-ranging (3-gain)* detector: every 16-bit raw word
carries its own gain stage in the top 2 bits and a 14-bit ADC code in the low
14 bits.  The three gain stages are the leading ``3`` axis of the
``pedestals`` / ``pixel_gain`` / ``pixel_offset`` constants.

Gain-bit decode (psana ``calib_jungfrau_single_panel`` + constants ``MSK``/``BSH``
at ``UtilsJungfrau.py`` ~lines 44-45, 652)::

    gbits = raw >> 14          # 0, 1, 2, or 3
    stage 0  <- gbits == 0
    stage 1  <- gbits == 1
    stage 2  <- gbits == 3     # NOTE: binary 11 (==3), NOT 2
    "bad"    <- gbits == 2     # binary 10 -> no gain stage; contributes 0
    adc = raw & 0x3fff
    calib[stage] = (adc - (pedestals + pixel_offset)[stage]) / pixel_gain[stage] * mask

Common-mode correction is OFF by default (matching psana's ``cmpars`` default
in this path being unused); this module does not apply it.  ``pixel_offset`` may
be absent for some runs/detectors -- callers pass ``None`` and it is treated as
0 (matching the snapshot provider's semantics).
"""

import numpy as np

from ._fastcalib import (                       # noqa: F401
    BACKEND as _BACKEND,
    calib_jungfrau_fast as _calib_jungfrau_fast,
    backend_info,
    check_out_buffer as _check_out_buffer,
    check_mask_shape as _check_mask_shape,
    # Re-exported ON PURPOSE (not used in this module): the constants-cache
    # contract that :func:`calib_jungfrau` documents below is unenforceable, so
    # its escape hatch and its diagnostics have to be reachable from the same
    # place a reader finds the contract.  Also re-exported from
    # :mod:`pscalib.apply` and from :mod:`pscalib`.
    memo_clear, memo_stats, memo_size, memo_nbytes, last_call,
)

#: 14-bit ADC mask -- psana ``UtilsJungfrau.MSK`` (``0x3fff``, ``(1<<14)-1``).
MSK = 0x3fff
#: Gain-bit shift -- psana ``UtilsJungfrau.BSH`` (the gain code is ``raw >> 14``).
BSH = 14

#: Expected number of gain stages on the leading constant axis (auto-ranging
#: 3-gain detector).
N_GAIN_STAGES = 3


def calib_jungfrau_reference(raw, pedestals, pixel_gain,
                             pixel_offset=None, mask=None, out=None):
    """The VERBATIM c5ce538 expression -- the byte-exactness oracle.

    This is the definition of correctness for :func:`calib_jungfrau`, kept in
    the tree (not in a scratch file) so the fast path can be cross-checked
    against it at any time, in-process, on real constants::

        ref  = calib_jungfrau_reference(raw, ped, gain, off, mask)
        fast = calib_jungfrau(raw, ped, gain, off, mask)
        assert (ref.view(np.uint32) == fast.view(np.uint32)).all()

    Do not "optimise" this function.  Every apparent redundancy in it is
    load-bearing: the ``np.select`` default lane gives the BAD gain code
    ``pedoff=0``/``factor=0`` so it COMPUTES ``(adc - 0.0) * 0.0`` (whose
    sign of zero is observable); the mask multiply is what turns a
    finite-negative masked pixel into ``-0.0``; and the three float32
    operations round THREE times, which a float64 chain would not.

    Original docstring follows.

    Calibrate a raw Jungfrau stack into ADU, fully offline (numpy only).

    Faithful re-implementation of psana
    ``UtilsJungfrau.calib_jungfrau`` / ``calib_jungfrau_single_panel`` (looped
    over segments).  Verified ``np.array_equal`` to ``det.raw.calib(evt)``.

    Parameters
    ----------
    raw : ndarray, shape ``(N, 512, 1024)``, uint16
        The raw detector stack (e.g. from :meth:`psdata.run.Event.stack`).
        ``N`` is the number of segments present this event.
    pedestals : ndarray, shape ``(3, S, 512, 1024)``, float32
        Per-(stage, segment) pedestals.  Leading axis = 3 gain stages.
    pixel_gain : ndarray, shape ``(3, S, 512, 1024)``, float32
        Per-(stage, segment) gain (ADU per keV-equivalent).  Calibration
        divides by this (protected: a 0 gain yields a 0 factor).
    pixel_offset : ndarray or None, shape ``(3, S, 512, 1024)``, float32
        Per-(stage, segment) pedestal offset, added to ``pedestals``.  ``None``
        is treated as 0 (matching psana / the snapshot, where it may be absent).
    mask : ndarray or None, shape ``(S, 512, 1024)``
        Per-segment status mask (0 = bad pixel).  ``None`` => no masking.
    out : ndarray or None, shape ``raw.shape``, float32
        OPTIONAL pre-allocated output buffer, so a per-event caller can reuse
        one 67 MB array instead of first-touching fresh pages every event.
        ``None`` (the default) allocates, exactly as before.  A supplied buffer
        is validated (ndarray / exact dtype / exact shape / writeable) and a
        mismatch RAISES -- it is never silently ignored (see
        :func:`pscalib.apply._fastcalib.check_out_buffer`).  The result is
        bit-identical either way: every element of the returned array is
        assigned by the segment loop below (``s`` runs over all of
        ``raw.shape[0]`` and each iteration assigns the whole segment), so the
        buffer's prior contents cannot survive anywhere.

    Returns
    -------
    ndarray, shape ``(N, 512, 1024)``, float32
        Calibrated stack in ADU.  Bad-gain-code pixels (``gbits == 2``) and
        masked pixels are 0.  This IS ``out`` when ``out`` was supplied.

    Notes
    -----
    ``raw`` carries ``N`` segments; the constants carry ``S`` segments
    (``S`` may exceed ``N`` if not every segment is present this event).  Each
    raw segment ``s`` indexes into constant segment ``s`` -- the caller must
    pass a ``raw`` stack ordered by ascending segment id (as
    :meth:`psdata.run.Event.stack` produces) and constants for those same
    segments.  For the reference run all 32 segments are always present.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(f"raw must be 3-D (N,512,1024); got shape {raw.shape}")

    pedestals = np.asarray(pedestals, dtype=np.float32)
    pixel_gain = np.asarray(pixel_gain, dtype=np.float32)
    if pedestals.shape[0] != N_GAIN_STAGES:
        raise ValueError(
            f"pedestals leading axis must be {N_GAIN_STAGES} gain stages; "
            f"got shape {pedestals.shape}")
    # A mask must be per-segment 3-D on BOTH routes, so that the two backends
    # agree about what a mask IS.  This validates, it does not compute: for every
    # mask of the documented shape the expression below is untouched.  (Without
    # it this route does not raise for a 2-D mask -- ``np.asarray(mask)[s]``
    # takes ROW s and broadcasts it over the whole segment, which is not masking.)
    if mask is not None:
        _check_mask_shape(mask, raw.shape)

    # poff = pedestals + pixel_offset  (offset absent -> 0)
    if pixel_offset is None:
        poff = pedestals.copy()
    else:
        poff = (pedestals + np.asarray(pixel_offset, dtype=np.float32)
                ).astype(np.float32)

    # gfac = 1 / pixel_gain, protected (gain==0 -> factor 0, matches psana)
    gfac = np.divide(1.0, pixel_gain,
                     out=np.zeros_like(pixel_gain, dtype=np.float32),
                     where=pixel_gain != 0).astype(np.float32)

    nseg = raw.shape[0]
    # ``np.zeros`` is kept for the allocating path so this stays the VERBATIM
    # c5ce538 expression.  A caller-supplied buffer is not zeroed -- and does
    # not need to be: the loop below assigns ``out[s]`` for every s in
    # range(raw.shape[0]), i.e. every element of the array, so no pre-existing
    # byte can reach the result.  (Validated first, never silently ignored.)
    if out is None:
        out = np.zeros(raw.shape, dtype=np.float32)
    else:
        out = _check_out_buffer(
            out, raw.shape,
            no_alias=(("raw", raw), ("pedestals", pedestals),
                      ("pixel_gain", pixel_gain),
                      ("pixel_offset", pixel_offset), ("mask", mask),
                      ("the derived poff", poff),
                      ("the derived gfac", gfac)))
    for s in range(nseg):
        arr = raw[s]                                    # (512,1024) uint16
        # gain bits: 00/01/11 select stages 0/1/2; 10 (==2) is the bad code.
        gbits = (arr >> BSH).astype(np.uint8)
        gr0, gr1, gr2 = gbits == 0, gbits == 1, gbits == 3
        factor = np.select((gr0, gr1, gr2),
                           (gfac[0, s], gfac[1, s], gfac[2, s]), default=0)
        pedoff = np.select((gr0, gr1, gr2),
                           (poff[0, s], poff[1, s], poff[2, s]), default=0)
        arrf = (arr & MSK).astype(np.float32)
        arrf -= pedoff
        arrf *= factor                                  # bad code -> factor 0
        if mask is not None:
            arrf *= np.asarray(mask)[s].astype(np.float32)
        out[s] = arrf
    return out


def calib_jungfrau(raw, pedestals, pixel_gain, pixel_offset=None, mask=None,
                   out=None):
    """Calibrate a raw Jungfrau stack into ADU, fully offline (numpy only).

    Byte-identical to :func:`calib_jungfrau_reference` (the verbatim c5ce538
    expression, kept above); this entry point merely evaluates it *faster*.  The
    signature, the return dtype/shape and every output bit -- including the sign
    of every zero and every NaN payload -- are unchanged.

    Where the speed comes from (see :mod:`pscalib.apply._fastcalib`):

    * ``poff = pedestals + pixel_offset`` and ``gfac = 1/pixel_gain`` are pure
      functions of the calibration constants, which are fetched once per run.
      The reference rebuilt both (two 201 MB arrays) on EVERY event; they are now
      HOISTED into an eviction-safe identity memo.
    * the reference materialised, per segment per event, a ``gbits`` array and
      TWO ``np.select`` outputs the size of the segment.  The fast path instead
      blocks the segment into spatial tiles and picks, per tile, the cheapest
      exact strategy (pure stage 0 / dense two-plane blend / stage-0 pass plus a
      sparse gather) from ONE cheap boolean pass over that tile -- the same pass
      whose ``count_nonzero`` makes the choice also IS the residual selector the
      gather consumes.
    * the mask multiply is pre-composed into the gain planes once per
      (``pixel_gain``, ``mask``) pair -- ``gfm = gfac * mask`` -- so the
      reference's third per-pixel multiply disappears from the per-event work.
      That fold is NOT an identity for floats, so it is taken only when
      ``pscalib.apply._fastcalib.fold_is_exact`` proves it (unsigned raw, finite
      and bounded ``poff``/``gfac``, and a mask that is exactly ``+0.0`` or
      ``1``); otherwise the unfolded path runs and still multiplies by the mask.
    * that tiled hybrid is the ONLY compute backend.  There is no JIT and no
      compiled kernel: ``import pscalib`` needs numpy and the python stdlib and
      nothing else.  ``PSCALIB_CALIB_BACKEND`` accepts ``auto`` / ``numpy`` /
      ``reference``; asking for anything else -- a compiled accelerator, say --
      is a hard ``ValueError``, never a silent fallback, so a measurement
      cannot be mislabelled.

    The tile choice depends only on the contents of the frame being calibrated,
    never on how events are grouped, so the result is invariant under any event
    partition; blocking is over SPATIAL axes only.

    ``out=`` (optional, default ``None`` == allocate, exactly as before) lets a
    per-event caller reuse ONE output buffer instead of first-touching 67 MB of
    fresh pages every event.  It is honoured on BOTH backends -- the tiled
    hybrid writes straight into it, and the ``reference`` backend fills it -- so
    ``PSCALIB_CALIB_BACKEND=reference`` keeps working with a buffer.  The output
    is BIT-IDENTICAL with and without it (every element is assigned either way);
    a buffer of the wrong dtype/shape RAISES rather than being ignored, as does
    one that may share memory with ``raw`` or with any of the constants.  Two
    things about it are the caller's problem: the returned array IS ``out``, so
    THE NEXT CALL OVERWRITES IT -- never hand the result to a queue, a thread or
    a deferred consumer without copying (``docs/fast-event-loops.md``) -- and
    exactly one buffer per concurrent caller.

    Constants contract (the memo)
    -----------------------------
    ``poff = pedestals + pixel_offset``, ``gfac = 1/pixel_gain`` and the
    ``gfm = gfac * mask`` fold are cached across calls, keyed on the IDENTITY of
    the arrays you pass.  That is where the speedup comes from, and it asks four
    things of the caller which the library cannot check for you:

    * **Do not mutate a constants array in place.**  Identity is not contents:
      ``pedestals[0] += 1`` between two calls does NOT invalidate the cache, and
      the next call is calibrated with the PRE-mutation constants (a measured max
      abs error of 15117 ADU on the reference dataset, silently equal to the
      previous answer).  Detecting it would mean content-hashing 201 MB per
      event.  Build a new array instead -- or call
      :func:`pscalib.apply.memo_clear` right after mutating, which is the
      supported escape hatch and always safe (a dropped entry is re-derived bit
      for bit).
    * **Pass ndarrays, and keep them alive.**  A ``list``, a ``tuple``, a python
      ``float`` or a numpy SCALAR cannot be weak-referenced, so it bypasses the
      cache entirely and is re-derived on EVERY event (measured: BYO lists cost
      ~1300 ms/event with zero amortisation).  ``pixel_offset`` must be an array
      or ``None`` -- ``pixel_offset=0.0`` is arithmetically the same and about
      2x the per-event cost, because ``None`` skips the derivation altogether.
      A provider that returns a FRESH array per event defeats the cache the same
      way: hold one constants set for the run.
    * **The cache is unbounded.**  Entries live until their source array dies.
      One jungfrau constants set holds 403 MB with ``pixel_offset=None``, 604 MB
      with an offset, up to 1074 MB if the constants arrive as float64; N live
      sets cost N times that.  :func:`pscalib.apply.memo_nbytes` reports it,
      :func:`pscalib.apply.memo_clear` drops it, and
      :func:`pscalib.apply.memo_stats` / :func:`pscalib.apply.memo_size` are the
      other two diagnostics.  The counters are ADVISORY (not atomic).  All four
      are re-exported at the top level too (``pscalib.memo_clear`` and friends),
      exactly as ``pscalib.calib_jungfrau`` is.
    * **Threading.**  Calling this from several threads is safe and produces
      identical bits; concurrent cold callers serialise inside the cache's miss
      path rather than each deriving their own copy.  See
      ``docs/fast-event-loops.md`` for the full threading contract of the
      recommended prefetch loop.

    ``mask=`` must be per-segment 3-D, ``(S, rows, cols)`` with ``S`` at least
    the segment count of ``raw``; a 2-D mask RAISES on both backends rather than
    being broadcast row-wise, which is what it would silently mean.

    See :func:`calib_jungfrau_reference` for the full parameter documentation.
    """
    if _BACKEND == "reference":
        return calib_jungfrau_reference(raw, pedestals, pixel_gain,
                                        pixel_offset=pixel_offset, mask=mask,
                                        out=out)
    return _calib_jungfrau_fast(raw, pedestals, pixel_gain,
                                pixel_offset=pixel_offset, mask=mask, out=out)


def gain_stage_map(raw):
    """Return the per-pixel gain *stage* (0/1/2) and a ``bad`` mask for ``raw``.

    Convenience/introspection helper (not used by :func:`calib_jungfrau`,
    which decodes inline).  ``stage`` is the index into the leading 3-axis of
    the gain constants; ``bad`` marks the ``gbits == 2`` code that has no stage.

    Returns
    -------
    (stage, bad) : (ndarray int8, ndarray bool), same shape as ``raw``
    """
    raw = np.asarray(raw)
    gbits = (raw >> BSH).astype(np.uint8)
    stage = np.full(raw.shape, -1, dtype=np.int8)
    stage[gbits == 0] = 0
    stage[gbits == 1] = 1
    stage[gbits == 3] = 2
    bad = gbits == 2
    return stage, bad
