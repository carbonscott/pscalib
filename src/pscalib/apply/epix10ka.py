"""pscalib.apply.epix10ka -- epix10ka 7-gain-range gain decode (vendored numpy).

The per-detector-type gain-decode leaf for the epix10ka family of multi-gain
detectors (epix10ka, epix10kaquad/-quad, epix10ka2m, ...).  A faithful,
framework-free numpy re-implementation of psana's
``psana.detector.UtilsEpix10ka.calib_epix10ka_any`` (and its helpers), verified
byte-identical (``np.array_equal``) to ``det.raw.calib(evt)`` for the reference
epixquad dataset (exp=ued1010667, run=177, det='epixquad').

This is the canonical home of the epix10ka decode -- it is the NEW (US-004)
sibling to :mod:`pscalib.apply.jungfrau`.

Unlike the Jungfrau (whose gain stage lives entirely in the raw word's top 2
bits), the epix10ka gain *range* of each pixel is decoded from TWO sources that
are OR-ed together (psana ``cbits_config_and_data_detector_alg``):

  1. The **per-ASIC Configure object** -- ``trbit`` (one trim bit per ASIC) and
     ``asicPixelConfig`` (a per-pixel 2-bit gain config).  These are *static*
     settings written into the Configure dgram, NOT calibration-DB constants;
     pscalib gets them from psdata's ``Run.seg_configs(detname)`` (US-003).

  2. The **per-event data gain bit** -- bit 14 of each raw word (``raw & B14``),
     shifted down to bit 5 of the control-bit word.

The OR of these gives a 6-bit control word per pixel; seven masks over it select
which of the seven gain ranges (``FH FM FL AHL_H AML_M AHL_L AML_L``) the pixel
sat in this event.  The calibration constants (``pedestals``, ``pixel_gain``)
carry a leading axis of 7 -- one (pedestal, gain) plane per gain range -- and
the per-event constant for each pixel is selected by its decoded range.

Calibration (psana ``calib_epix10ka_any``)::

    cbits = cbits_config(trbit, asicPixelConfig)        # per-ASIC static bits
    cbits = cbits_config | ((raw & B14) >> 9)           # + per-event data bit
    gmaps = gain_maps(cbits)                            # 7 boolean range masks
    factor  = select(gmaps, 1/pixel_gain[range], default=1)
    pedest  = select(gmaps, pedestals[range],   default=0)
    calib   = (raw & 0x3fff - pedest) * factor * mask

Common-mode correction is OFF by default (matching psana's default ``cmpars``
path being unused for this dataset); this module does not apply it.  There is
deliberately no compiled code (the only C++ kernel in the calib engine is
Jungfrau's optional ``cversion=3`` speed twin, irrelevant here).
"""

import numpy as np

# --------------------------------------------------------------------------
# Bit constants -- vendored verbatim from psana UtilsEpix10ka.py (lines 56-64).
# --------------------------------------------------------------------------
#: trbit control bit -- ``1<<4`` (the 5th bit), OR-ed in per ASIC.
B04 = 0o20      # 16
#: data-gain control bit slot -- ``1<<5`` (the 6th bit), where the per-event
#: gain bit lands after the down-shift.
B05 = 0o40      # 32
#: epix10ka data gain bit -- ``1<<14`` (the 15th bit of the raw word).
B14 = 0o40000   # 16384
#: epix10ka 14-bit ADC data mask -- ``(1<<14)-1``.
M14 = 0x3fff    # 16383
#: shift moving the data gain bit (B14) down into the B05 control-bit slot
#: (``14 - 5 == 9``).  psana ``epix_base._gain_bit_shift``.
GAIN_BIT_SHIFT = 9

#: Number of gain ranges (leading axis of the epix10ka constants).
N_GAIN_RANGES = 7

#: psana ``UtilsEpix10ka.GAIN_MODES`` -- the seven gain-range names, in the same
#: order as the leading axis of the constants and the gain maps below.
GAIN_MODES = ("FH", "FM", "FL", "AHL_H", "AML_M", "AHL_L", "AML_L")

#: Per-segment panel shape for an epix10ka ASIC quad (2x2 ASICs of 176x192).
PANEL_SHAPE = (352, 384)


# ==========================================================================
# Per-ASIC static control bits from the Configure object
# ==========================================================================
def cbits_config_epix10ka(trbit, asic_pixel_config, shape=PANEL_SHAPE):
    """Per-segment control bits ``(352, 384)`` from one panel's Configure fields.

    Faithful re-implementation of psana
    ``UtilsEpix10ka.cbits_config_epix10ka`` (which takes the segment config
    object ``cob`` and reads ``cob.trbit`` / ``cob.asicPixelConfig``).  Here the
    two arrays are passed explicitly (pscalib gets them from psdata's
    ``seg_configs``), keeping this function psana-free.

    The four ASICs (each ``176x192``) are reassembled into the ``352x384`` panel
    in psana's exact orientation -- ASICs 2 and 1 (top row) are flipped in both
    axes; ASICs 3 and 0 (bottom row) are placed as-is -- then masked to the two
    gain config bits (``& 12``) and OR-ed with the trbit (``B04``) per ASIC.

    Parameters
    ----------
    trbit : array-like, shape ``(4,)``
        Per-ASIC trim bit (``cob.trbit``), one entry per ASIC.
    asic_pixel_config : ndarray, shape ``(4, 176, 192)``, uint8
        Per-ASIC per-pixel gain config (``cob.asicPixelConfig``).
    shape : (int, int)
        Panel shape; ``(352, 384)`` for epix10ka.

    Returns
    -------
    ndarray, shape ``(352, 384)``, uint8
        The static (config-only) per-pixel control bits.
    """
    trbits = np.asarray(trbit)
    pca = np.asarray(asic_pixel_config)
    rowsh, colsh = shape[0] // 2, shape[1] // 2     # 176, 192

    # Reassemble the 4 ASICs into the panel (psana orientation):
    #   top row    = [flip(flip(A2)), flip(flip(A1))]
    #   bottom row = [A3,             A0           ]
    cbits = np.vstack((
        np.hstack((np.flipud(np.fliplr(pca[2])), np.flipud(np.fliplr(pca[1])))),
        np.hstack((pca[3], pca[0])),
    ))
    # keep only the two gain config bits (0b1100 == 12).
    np.bitwise_and(cbits, 12, out=cbits)

    # OR in the trbit (B04) per ASIC -- psana's exact per-quadrant mapping.
    if all(trbits):
        cbits = np.bitwise_or(cbits, B04)
    elif not any(trbits):
        return cbits
    else:
        if trbits[2]:
            np.bitwise_or(cbits[:rowsh, :colsh], B04, out=cbits[:rowsh, :colsh])
        if trbits[3]:
            np.bitwise_or(cbits[rowsh:, :colsh], B04, out=cbits[rowsh:, :colsh])
        if trbits[0]:
            np.bitwise_or(cbits[rowsh:, colsh:], B04, out=cbits[rowsh:, colsh:])
        if trbits[1]:
            np.bitwise_or(cbits[:rowsh, colsh:], B04, out=cbits[:rowsh, colsh:])
    return cbits


def cbits_config_detector(seg_configs, segment_ids=None, shape=PANEL_SHAPE):
    """Stack per-segment static control bits into ``(n_segments, 352, 384)``.

    psana ``epix_base._cbits_config_detector`` stacks
    :func:`cbits_config_epix10ka` over the detector's segments in segment-id
    order.  ``seg_configs`` is psdata's ``Run.seg_configs(detname)`` mapping
    ``{segment_id: seg_cfg}`` where ``seg_cfg.config.{trbit,asicPixelConfig}``
    expose the two fields (byte-identical to psana
    ``det.raw._seg_configs()[seg].config.{trbit,asicPixelConfig}``).

    Parameters
    ----------
    seg_configs : dict
        ``{segment_id: seg_cfg}`` (psdata ``seg_configs``).
    segment_ids : sequence of int or None
        Segments to stack, in order.  ``None`` (default) uses every segment in
        ``seg_configs``, sorted ascending -- matching psana's ``dcfg.items()``
        order and the raw stack ordering of :meth:`psdata.stream.Event.stack`.
    shape : (int, int)
        Panel shape.

    Returns
    -------
    ndarray, shape ``(n_segments, 352, 384)``, uint8
    """
    seg_ids = sorted(seg_configs) if segment_ids is None else list(segment_ids)
    planes = []
    for s in seg_ids:
        cfg = seg_configs[s].config
        planes.append(cbits_config_epix10ka(cfg.trbit, cfg.asicPixelConfig,
                                            shape=shape))
    return np.stack(tuple(planes))


def cbits_config_and_data(raw, cbits_config, data_gain_bit=B14,
                          gain_bit_shift=GAIN_BIT_SHIFT):
    """Add the per-event data gain bit to the static control bits.

    Faithful re-implementation of psana
    ``UtilsEpix10ka.cbits_config_and_data_detector_alg``: extract bit 14 of each
    raw word (``raw & B14``), shift it down into the B05 control-bit slot, and
    OR it into a *copy* of the config control bits (never mutating the input).

    Parameters
    ----------
    raw : ndarray, uint16, shape ``(n_segments, 352, 384)``
        The raw detector stack this event.
    cbits_config : ndarray, shape ``(n_segments, 352, 384)``
        The static per-pixel control bits from :func:`cbits_config_detector`.
    data_gain_bit : int
        The raw-word gain bit (``B14`` for epix10ka).
    gain_bit_shift : int
        Down-shift from the gain bit to the B05 slot (9 for epix10ka).

    Returns
    -------
    ndarray, same shape as ``cbits_config``
        The combined config+data control bits.
    """
    if cbits_config is None:
        return None
    if raw is None:
        return cbits_config
    datagainbit = np.bitwise_and(raw, data_gain_bit)
    databit05 = np.right_shift(datagainbit, gain_bit_shift)   # B14 -> B05
    return np.bitwise_or(cbits_config, databit05)             # copy, no mutate


# ==========================================================================
# Gain-range maps and per-event constant selection
# ==========================================================================
def gain_maps_epix10ka_any(cbits):
    """Seven boolean gain-range maps from the combined control bits.

    Faithful re-implementation of psana
    ``UtilsEpix10ka.gain_maps_epix10ka_any_alg``.  Returns a 7-tuple of boolean
    arrays (shaped as ``cbits``), one per gain range, in ``GAIN_MODES`` order::

        FH=0  FM=1  FL=2  AHL_H=3  AML_M=4  AHL_L=5  AML_L=6

    Parameters
    ----------
    cbits : ndarray
        Combined config+data control bits
        (:func:`cbits_config_and_data`).

    Returns
    -------
    tuple of 7 ndarray(bool)
    """
    if cbits is None:
        return None
    cbits_m60 = cbits & 60   # 0b111100
    cbits_m28 = cbits & 28   # 0b011100
    cbits_m12 = cbits & 12   # 0b001100
    return ((cbits_m28 == 28),   # FH
            (cbits_m28 == 12),   # FM
            (cbits_m12 == 8),    # FL
            (cbits_m60 == 16),   # AHL_H
            (cbits_m60 == 0),    # AML_M
            (cbits_m60 == 48),   # AHL_L
            (cbits_m60 == 32))   # AML_L


def event_constants_for_gmaps(gmaps, cons, default=0):
    """Per-event per-pixel constant selected by gain range.

    Faithful re-implementation of psana
    ``UtilsEpix10ka.event_constants_for_gmaps``: an ``np.select`` of the seven
    gain-range planes of ``cons`` driven by the seven boolean ``gmaps``.

    Parameters
    ----------
    gmaps : tuple of 7 ndarray(bool), each ``(n_segments, 352, 384)``
        The gain-range masks (:func:`gain_maps_epix10ka_any`).
    cons : ndarray, shape ``(7, n_segments, 352, 384)``
        A 7-gain-range constant (e.g. pedestals or gain factors).
    default : scalar
        Value where no gain range matched.

    Returns
    -------
    ndarray, shape ``(n_segments, 352, 384)``
    """
    if gmaps is None or cons is None:
        return None
    return np.select(gmaps,
                     (cons[0], cons[1], cons[2], cons[3],
                      cons[4], cons[5], cons[6]),
                     default=default)


def gain_factor_from_gain(pixel_gain):
    """``1 / pixel_gain`` with division-by-zero protected (gain 0 -> factor 0).

    Matches psana ``NDArrUtils.divide_protected(ones_like(gain), gain)`` used to
    build ``store.gfac``.
    """
    gain = np.asarray(pixel_gain, dtype=np.float32)
    return np.divide(np.ones_like(gain), gain,
                     out=np.zeros_like(gain, dtype=np.float32),
                     where=gain != 0).astype(np.float32)


# ==========================================================================
# The calib entry point
# ==========================================================================
def calib_epix10ka(raw, pedestals, pixel_gain, seg_configs,
                   mask=None, segment_ids=None,
                   data_bit_mask=M14, data_gain_bit=B14,
                   gain_bit_shift=GAIN_BIT_SHIFT):
    """Calibrate a raw epix10ka stack into ADU, fully offline (numpy only).

    Faithful re-implementation of psana ``UtilsEpix10ka.calib_epix10ka_any``
    (with common-mode OFF, matching its default path for this dataset).
    Verified ``np.array_equal`` to ``det.raw.calib(evt)``.

    Parameters
    ----------
    raw : ndarray, shape ``(n_segments, 352, 384)``, uint16
        The raw detector stack (e.g. from :meth:`psdata.stream.Event.stack`),
        ordered by ascending segment id.
    pedestals : ndarray, shape ``(7, n_segments, 352, 384)``, float32
        Per-(gain-range, segment) pedestals.  Leading axis = 7 gain ranges.
    pixel_gain : ndarray, shape ``(7, n_segments, 352, 384)``, float32
        Per-(gain-range, segment) gain (ADU per keV-equivalent).  Calibration
        divides by this (protected: a 0 gain yields a 0 factor).
    seg_configs : dict
        ``{segment_id: seg_cfg}`` from psdata's ``Run.seg_configs(detname)``;
        each ``seg_cfg.config`` exposes ``trbit`` ``(4,)`` and
        ``asicPixelConfig`` ``(4,176,192)``.  This is the load-bearing per-ASIC
        config that drives the gain-range decode (it is NOT in the calib DB).
    mask : ndarray or None, shape ``(n_segments, 352, 384)``
        Per-segment good/bad (1/0) mask.  ``None`` => no masking.  To reproduce
        ``det.raw.calib(evt)`` byte-exactly pass the default status mask --
        either the snapshot's cached ``mask`` (``det.raw._mask()``) or one built
        from ``pixel_status`` via :func:`mask_from_pixel_status`.
    segment_ids : sequence of int or None
        Segment order to use for the config stack; ``None`` => sorted segments
        of ``seg_configs`` (matching the raw stack order).
    data_bit_mask, data_gain_bit, gain_bit_shift : int
        Detector-family bit parameters; the defaults are epix10ka's
        (``0x3fff`` / ``1<<14`` / ``9``).

    Returns
    -------
    ndarray, shape ``(n_segments, 352, 384)``, float32
        Calibrated stack in ADU.

    Notes
    -----
    ``raw`` carries the segments present this event; ``seg_configs`` /
    ``pedestals`` / ``pixel_gain`` must cover those same segments in the same
    order.  For the reference run all 4 segments are always present.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(
            f"raw must be 3-D (n_segments,352,384); got shape {raw.shape}")

    pedestals = np.asarray(pedestals, dtype=np.float32)
    pixel_gain = np.asarray(pixel_gain, dtype=np.float32)
    for name, arr in (("pedestals", pedestals), ("pixel_gain", pixel_gain)):
        if arr.shape[0] != N_GAIN_RANGES:
            raise ValueError(
                f"{name} leading axis must be {N_GAIN_RANGES} gain ranges; "
                f"got shape {arr.shape}")

    # static config control bits, then + per-event data gain bit
    cbits_cfg = cbits_config_detector(seg_configs, segment_ids=segment_ids)
    cbits = cbits_config_and_data(raw, cbits_cfg,
                                  data_gain_bit=data_gain_bit,
                                  gain_bit_shift=gain_bit_shift)

    # 7 gain-range masks; select per-event pedestal + gain factor
    gmaps = gain_maps_epix10ka_any(cbits)
    gfac = gain_factor_from_gain(pixel_gain)
    factor = event_constants_for_gmaps(gmaps, gfac, default=1)
    pedest = event_constants_for_gmaps(gmaps, pedestals, default=0)

    # (code - pedestal) * gain  [* mask]
    arrf = np.array(raw & data_bit_mask, dtype=np.float32)
    if pedest is not None:
        arrf -= pedest
    out = arrf * factor
    if mask is not None:
        out = out * np.asarray(mask)
    return out.astype(np.float32)


# ==========================================================================
# Default status mask (for the BYO / web path that has no cached mask)
# ==========================================================================
def status_as_mask(status, status_bits=(1 << 64) - 1):
    """Good/bad (1/0) mask from a ``pixel_status`` array.

    Faithful re-implementation of psana ``UtilsMask.status_as_mask``: a pixel is
    good (1) iff none of the ``status_bits`` are set in its status word.
    """
    status = np.asarray(status)
    cond = (status & status_bits) > 0
    return np.asarray(np.select((cond,), (0,), default=1), dtype=np.uint8)


def merge_mask_for_grinds(mask, gain_range_inds=(0, 1, 2, 3, 4)):
    """Merge a per-gain-range mask over gain ranges (logical AND).

    Faithful re-implementation of psana ``UtilsMask.merge_mask_for_grinds``: for
    a 4-D mask ``(7, n_segments, 352, 384)`` AND-merges the requested gain-range
    planes down to ``(n_segments, 352, 384)``.  Lower-rank masks pass through.
    epix10ka uses ``(0,1,2,3,4)`` (the five configured ranges; the two evaluated
    ranges 5/6 are excluded -- psana caps grinds at 5).
    """
    mask = np.asarray(mask)
    if mask.ndim < 4:
        return mask
    m = mask.astype(np.uint8)
    out = np.copy(m[gain_range_inds[0]])
    for i in gain_range_inds[1:]:
        if i < mask.shape[0]:
            cond = np.logical_and(out, m[i])
            out = np.asarray(np.select((cond,), (1,), default=0),
                             dtype=np.uint8)
    return out


def mask_from_pixel_status(pixel_status, status_bits=(1 << 64) - 1,
                           gain_range_inds=(0, 1, 2, 3, 4)):
    """Build the default epix10ka calib mask from ``pixel_status``.

    Reproduces psana's default ``det.raw._mask()`` for epix10ka (which
    ``det.raw.calib(evt)`` uses): :func:`status_as_mask` over the full status
    bitword, then :func:`merge_mask_for_grinds` over gain ranges ``(0,1,2,3,4)``.
    Use this on the BYO / web retrieval path (where no cached mask was
    snapshotted) to get a byte-exact result.

    Parameters
    ----------
    pixel_status : ndarray, shape ``(7, n_segments, 352, 384)``, uint64
        The ``pixel_status`` calibration constant.
    status_bits : int
        Status bits to treat as "bad" (default: all 64).
    gain_range_inds : sequence of int
        Gain ranges to merge (default epix10ka's ``(0,1,2,3,4)``).

    Returns
    -------
    ndarray, shape ``(n_segments, 352, 384)``, uint8
    """
    smask = status_as_mask(np.asarray(pixel_status).astype(np.uint64),
                           status_bits=status_bits)
    return merge_mask_for_grinds(smask, gain_range_inds=gain_range_inds)
