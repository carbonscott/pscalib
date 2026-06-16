"""pscalib.registry -- detector-type -> apply-plugin dispatch (the thin seam).

The agreed thin abstraction: a detector *plugin* is just a function

    plugin(raw, constants, config=None) -> calib

registered by detector type.  No class hierarchy -- jungfrau and epix10ka are
two leaf plugins over the pure-numpy gain decoders in :mod:`pscalib.apply`.

  * ``constants`` is a uniform mapping ``{ctype: ndarray}`` (a plain dict, a
    :class:`pscalib.providers.snapshot.CalibSnapshot`, or any object exposing
    ``.array(ctype)`` / ``__getitem__``).  It is the *constants contract*: the
    plugin pulls the ctypes it needs (``pedestals``, ``pixel_gain``, ...).
  * ``config`` is the per-segment Configure object the detector needs to decode
    its gain (psdata's ``Run.seg_configs(detname)``).  jungfrau ignores it (its
    gain is in the raw bits); epix10ka *requires* it.

This module is the registry US-005's unified ``pscalib.calib(raw, constants,
config=None)`` dispatches through; US-004 lands it with the jungfrau + epix10ka
leaves so both share one dispatch.  Pure numpy -- importing it pulls in only
numpy (the apply leaves it imports are numpy-only).
"""

import numpy as np

from .apply.jungfrau import calib_jungfrau, N_GAIN_STAGES as _JF_STAGES
from .apply.epix10ka import calib_epix10ka, mask_from_pixel_status

__all__ = [
    "register", "get_plugin", "registered_types", "calib",
    "plugin_jungfrau", "plugin_epix10ka",
    "detector_type_of", "detector_type_for_constants",
]

#: detector-type (str) -> plugin function.  Populated at import time with the
#: built-in leaves (see the ``register`` calls at the bottom).
_REGISTRY = {}


def register(det_type, plugin):
    """Register ``plugin`` (a ``plugin(raw, constants, config=None) -> calib``
    callable) for detector type ``det_type``.  Returns ``plugin`` so it can be
    used as a decorator."""
    if not callable(plugin):
        raise TypeError(f"plugin for {det_type!r} must be callable")
    _REGISTRY[det_type] = plugin
    return plugin


def get_plugin(det_type):
    """Return the registered plugin for ``det_type``.

    ``det_type`` may be a bare type (``"epix10ka"``) or a psana drp class name
    (``"epix10ka_raw_2_0_1"``); the leading family token is matched, so both
    resolve to the epix10ka plugin.  Raises ``KeyError`` if unknown.
    """
    norm = detector_type_of(det_type)
    if norm in _REGISTRY:
        return _REGISTRY[norm]
    raise KeyError(
        f"no apply plugin registered for detector type {det_type!r} "
        f"(known: {registered_types()})")


def registered_types():
    """Sorted list of registered detector types."""
    return sorted(_REGISTRY)


def detector_type_of(det_type):
    """Normalize a detector type/class name to its registered family token.

    psana detector class names look like ``epix10ka_raw_2_0_1`` or
    ``jungfrau_raw_0_1_0``; the family is the leading token before the first
    ``_raw`` / version suffix.  An already-bare ``"epix10ka"`` passes through.
    Any ``epix10ka*`` / ``epixquad`` family name maps to ``"epix10ka"``.
    """
    s = str(det_type)
    # strip a psana "<family>_raw_x_y_z" suffix to the family token
    for sep in ("_raw_", "_raw"):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    s = s.lower()
    # epix10ka composite/family aliases all decode the same way
    if s.startswith("epix10ka") or s in ("epixquad", "epix10kaquad",
                                         "epix10ka2m"):
        return "epix10ka"
    return s


def detector_type_for_constants(constants):
    """Infer the registered detector type from the *constants alone* (US-005).

    The US-005 public surface (:func:`calib` called as
    ``calib(raw, constants, config=None)``) takes no ``det_type`` argument; it
    recovers the detector family from the constants themselves -- a snapshot's
    ``detname`` / per-ctype ``dettype`` metadata, a web fetch's metadata docs, or
    an explicit naming key on a BYO dict (see
    :func:`pscalib.model.detector_type_hint`) -- then normalizes it via
    :func:`detector_type_of`.

    Returns the normalized family token (e.g. ``"jungfrau"``, ``"epix10ka"``).
    Raises ``ValueError`` if the constants carry no recoverable detector
    identity or it does not map to a registered plugin -- in which case the
    caller must pass ``det_type`` explicitly.
    """
    from .model import detector_type_hint
    hint = detector_type_hint(constants)
    if hint is None:
        raise ValueError(
            "could not infer detector type from constants (no dettype/detname "
            "metadata and no naming key); call calib(det_type, raw, constants, "
            "...) with an explicit det_type, or pass constants that carry their "
            "detector identity (a snapshot / web fetch)")
    norm = detector_type_of(hint)
    if norm not in _REGISTRY:
        raise ValueError(
            f"constants name detector type {hint!r} (normalized {norm!r}) which "
            f"has no registered apply plugin (known: {registered_types()}); "
            f"call calib(det_type, raw, constants, ...) with an explicit "
            f"det_type")
    return norm


# ==========================================================================
# Constants-contract access helper
# ==========================================================================
def _get_const(constants, ctype, required=True):
    """Pull ``ctype`` from a uniform constants mapping.

    Accepts a plain ``{ctype: ndarray}`` dict, a
    :class:`~pscalib.providers.snapshot.CalibSnapshot` (``.array(ctype)`` /
    ``.mask``), or a psana-style ``{ctype: (ndarray, meta)}`` dict.  Returns the
    ndarray (or ``None`` when ``required`` is False and absent).
    """
    val = None
    if hasattr(constants, "array") and callable(constants.array):
        # CalibSnapshot-like
        if ctype == "mask":
            val = getattr(constants, "mask", None)
        else:
            val = constants.array(ctype)
    elif hasattr(constants, "get"):
        val = constants.get(ctype)
    else:
        try:
            val = constants[ctype]
        except (KeyError, TypeError, IndexError):
            val = None
    # unwrap psana-style (ndarray, meta) tuples
    if isinstance(val, (tuple, list)) and val and isinstance(val[0], np.ndarray):
        val = val[0]
    if val is None and required:
        raise KeyError(
            f"constants are missing required ctype {ctype!r}")
    return val


# ==========================================================================
# Built-in plugins (the thin seam): plugin(raw, constants, config=None) -> calib
# ==========================================================================
def plugin_jungfrau(raw, constants, config=None):
    """Jungfrau apply plugin -- gain stage is in the raw bits, ``config`` unused.

    Pulls ``pedestals`` / ``pixel_gain`` (+ optional ``pixel_offset`` / ``mask``)
    from the constants mapping and runs :func:`pscalib.apply.calib_jungfrau`.

    The mask: if the constants carry a cached ``mask`` (a snapshot's
    ``det.raw._mask()``) it is used; otherwise, if ``pixel_status`` is present,
    the default status mask is derived (:func:`mask_from_pixel_status`, whose
    gain-range merge clamps to jungfrau's three ranges) so the BYO / web path is
    byte-exact too -- psana's ``det.raw.calib(evt)`` masks bad pixels, so a
    web/BYO apply that skipped masking would differ.  If neither is available,
    no mask is applied.
    """
    pedestals = _get_const(constants, "pedestals")
    pixel_gain = _get_const(constants, "pixel_gain")
    pixel_offset = _get_const(constants, "pixel_offset", required=False)
    mask = _get_const(constants, "mask", required=False)
    if mask is None:
        status = _get_const(constants, "pixel_status", required=False)
        if status is not None:
            mask = mask_from_pixel_status(status)
    return calib_jungfrau(raw, pedestals, pixel_gain,
                          pixel_offset=pixel_offset, mask=mask)


def plugin_epix10ka(raw, constants, config=None):
    """epix10ka apply plugin -- ``config`` (per-segment Configure) is REQUIRED.

    Pulls ``pedestals`` / ``pixel_gain`` from the constants mapping and the
    per-ASIC ``trbit`` / ``asicPixelConfig`` from ``config`` (psdata's
    ``seg_configs``), then runs :func:`pscalib.apply.calib_epix10ka`.

    The mask: if the constants carry a cached ``mask`` (a snapshot's
    ``det.raw._mask()``) it is used; otherwise, if ``pixel_status`` is present,
    the default status mask is derived (:func:`mask_from_pixel_status`) so the
    BYO / web path is byte-exact too.  If neither is available, no mask is
    applied.
    """
    if config is None:
        raise ValueError(
            "epix10ka apply requires the per-segment Configure object "
            "(config=run.seg_configs(detname)); it drives the gain-range decode "
            "and is not in the calib DB")
    pedestals = _get_const(constants, "pedestals")
    pixel_gain = _get_const(constants, "pixel_gain")
    mask = _get_const(constants, "mask", required=False)
    if mask is None:
        status = _get_const(constants, "pixel_status", required=False)
        if status is not None:
            mask = mask_from_pixel_status(status)
    return calib_epix10ka(raw, pedestals, pixel_gain, config, mask=mask)


# register the built-in leaves under one dispatch
register("jungfrau", plugin_jungfrau)
register("epix10ka", plugin_epix10ka)


# ==========================================================================
# Unified public entry point (US-005)
# ==========================================================================
def _enforce_validity(constants, run, allow_stale, log):
    """Run the US-002 refuse-by-default staleness check, if a run is given.

    No-op when ``run is None`` (the US-000/US-004 byte-exact gates call
    ``calib(raw, constants)`` with no run -- preserving their numbers).  When a
    run is given, derives ``{ctype: Validity}`` from the constants and delegates
    to :func:`pscalib.model.check_validity`: in range -> silent, out of range ->
    raises ``StaleConstantsError`` unless ``allow_stale`` (then a warning).
    """
    if run is None:
        return
    from .model import (Constants, check_validity, validities_from_calibconst)
    c = constants if isinstance(constants, Constants) else None
    if c is not None:
        validities = c.validities()
        pin = c.pin
    elif hasattr(constants, "validities") and callable(constants.validities):
        validities = constants.validities()
        pin = getattr(constants, "pin_obj", None)
    elif hasattr(constants, "calibconst") and callable(constants.calibconst):
        validities = validities_from_calibconst(constants.calibconst())
        pin = getattr(constants, "pin_obj", None)
    elif hasattr(constants, "items"):
        validities = validities_from_calibconst(constants)
        pin = None
    else:
        validities = {}
        pin = None
    check_validity(validities, run, allow_stale=allow_stale, pin=pin, log=log)


def calib(*args, config=None, run=None, allow_stale=False, log=None):
    """Apply calibration constants to ``raw`` in pure numpy -- the public surface.

    Two call forms share this one entry point (and one registry dispatch):

    **Inferred (US-005, preferred)** -- ``calib(raw, constants, config=None)``::

        out = pscalib.calib(raw, snap, config=seg_cfg)   # det_type inferred

    The detector type is recovered from the constants themselves (a snapshot's
    ``detname``/``dettype``, a web fetch's metadata, or a BYO dict's naming key;
    see :func:`detector_type_for_constants`).

    **Explicit (US-004, legacy)** -- ``calib(det_type, raw, constants,
    config=None)``::

        out = pscalib.calib("epix10ka_raw_2_0_1", raw, snap, config=seg_cfg)

    A leading ``str`` first argument is taken as ``det_type``; anything else is
    taken as ``raw`` and the type is inferred.  Both forms route through
    :func:`get_plugin` to the same ``plugin(raw, constants, config=None) ->
    calib`` leaf.

    Validity enforcement (US-002) is wired in: pass ``run=`` to enforce that the
    constants are valid for that run *before* applying -- out of range raises
    :class:`pscalib.model.StaleConstantsError` by default, ``allow_stale=True``
    downgrades to a warning, in range is silent.  With no ``run`` the check is
    skipped (preserving the US-000/US-004 byte-exact numbers).

    Parameters
    ----------
    *args
        Either ``(raw, constants)`` (inferred) or ``(det_type, raw,
        constants)`` (explicit).
    config : object, optional
        The per-segment Configure object some detectors need (epix10ka requires
        it; jungfrau ignores it).
    run : int, optional
        The run being calibrated; enables US-002 staleness enforcement.
    allow_stale : bool
        Downgrade an out-of-range refusal to a logged warning.
    log : logging.Logger, optional
        Logger for the staleness warning.

    Returns
    -------
    numpy.ndarray
        The calibrated stack (byte-exact vs ``det.raw.calib(evt)``).
    """
    if args and isinstance(args[0], str):
        # explicit form: calib(det_type, raw, constants)
        if len(args) != 3:
            raise TypeError(
                "calib(det_type, raw, constants, config=..., run=...) takes a "
                f"det_type, raw and constants; got {len(args)} positional args")
        det_type, raw, constants = args
        norm = detector_type_of(det_type)
    else:
        # inferred form: calib(raw, constants)
        if len(args) != 2:
            raise TypeError(
                "calib(raw, constants, config=..., run=...) takes raw and "
                f"constants; got {len(args)} positional args (for an explicit "
                "detector type use calib(det_type, raw, constants, ...))")
        raw, constants = args
        norm = detector_type_for_constants(constants)

    _enforce_validity(constants, run, allow_stale, log)
    return get_plugin(norm)(raw, constants, config=config)
