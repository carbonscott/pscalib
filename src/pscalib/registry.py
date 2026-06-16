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
    "detector_type_of",
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
    """
    pedestals = _get_const(constants, "pedestals")
    pixel_gain = _get_const(constants, "pixel_gain")
    pixel_offset = _get_const(constants, "pixel_offset", required=False)
    mask = _get_const(constants, "mask", required=False)
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
# Unified entry point (US-005 builds the public pscalib.calib on top of this)
# ==========================================================================
def calib(det_type, raw, constants, config=None):
    """Dispatch ``raw`` + ``constants`` to ``det_type``'s plugin and return the
    calibrated stack.

    The registry-level dispatch: ``det_type`` selects the plugin (jungfrau or
    epix10ka today), which pulls the ctypes it needs from the uniform
    ``constants`` mapping and applies them in pure numpy, byte-exact vs
    ``det.raw.calib(evt)``.
    """
    return get_plugin(det_type)(raw, constants, config=config)
