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

import os

import numpy as np

from .apply.jungfrau import calib_jungfrau, N_GAIN_STAGES as _JF_STAGES
from .apply.epix10ka import calib_epix10ka, mask_from_pixel_status

__all__ = [
    "register", "get_plugin", "registered_types", "calib",
    "plugin_jungfrau", "plugin_epix10ka",
    "detector_type_of", "detector_type_for_constants",
    "UnvalidatedCalibVersionError", "validated_versions",
    "ConstantsRawMismatchError",
]

#: detector-type (str) -> plugin function.  Populated at import time with the
#: built-in leaves (see the ``register`` calls at the bottom).
_REGISTRY = {}


# ==========================================================================
# CAL-15: version-aware dispatch guard ("know what you don't know")
# ==========================================================================
# The bug (CAL-15): pscalib dispatches on the detector FAMILY token
# (``epix10ka_raw_3_0_1`` -> ``epix10ka``), but *psana* selects the calib
# ALGORITHM from the full VERSION TRIPLE -- the deployed release ships two
# epix10ka calib implementations (``calib_epix10ka_v02`` for the newer class,
# the legacy ``calib_epix10ka_any`` for classes that don't override).  Raw
# decode is self-describing; calibration is NOT -- the algorithm is chosen by a
# token the old ``detector_type_of`` throws away.  pscalib has ONE plugin per
# family, so applying it to a version it was never byte-exactness-tested against
# is a *silent guess*.
#
# This module cannot grow a second epix10ka implementation (that is a feature,
# out of scope).  What it MUST do is stop guessing silently: keep the family
# normalization for LOOKUP, but ALSO record which full version triples the
# single family plugin has actually been validated against, and REFUSE (by
# default) when handed an unknown/unvalidated version rather than applying the
# plugin with no signal at all.  Same class of fix as psdata's DET-10.
#
# Seeding -- the version triples below are the ONLY ones this repo's
# byte-exactness oracle has actually measured pscalib's single plugin against:
#
#   * ``epix10ka_raw_2_0_1`` -- the class of ``det.raw`` in the epix10ka
#     byte-exact gate (exp=ued1010667 run=177 det=epixquad; see
#     ``tests/test_epix10ka_us004.py`` -- ``det_type = type(det.raw).__name__``
#     -- and the explicit synthetic calls in test_epix10ka_us004 /
#     test_epix10ka_trbit_us008).  ``max|diff| == 0`` vs ``det.raw.calib(evt)``.
#   * ``jungfrau_raw_0_1_0`` -- the jungfrau raw class of the byte-exact render
#     gate (exp=mfx100848724 run=51 det=jungfrau; see
#     ``tests/test_calib_us000.py`` and the explicit call in
#     ``tests/test_purity_us007.py``).  ``max|diff| == 0`` vs
#     ``det.raw.calib(evt)``.
#
# UNCERTAINTY (documented deliberately): the repo records the *class names*
# above but does NOT record which psana calib FUNCTION produced each ground
# truth -- that is the very information CAL-15 says was never captured.  So this
# set is deliberately CONSERVATIVE: it lists only versions with a recorded
# byte-exact gate.  A genuinely-new deployed class (e.g. ``epix10ka_raw_3_0_1``,
# which psana routes to ``calib_epix10ka_v02``) is intentionally ABSENT -- it
# must be proven byte-exact and added here explicitly, never guessed at.
#: family token -> frozenset of validated psana drp class names (lowercase).
_VALIDATED_VERSIONS = {
    "epix10ka": frozenset({"epix10ka_raw_2_0_1"}),
    "jungfrau": frozenset({"jungfrau_raw_0_1_0"}),
}

#: env escape hatch: set to 1/true/yes/on to allow an unvalidated version
#: through (mirrors the ``allow_unvalidated=True`` opt-in kwarg on :func:`calib`).
_ENV_ALLOW_UNVALIDATED = "PSCALIB_ALLOW_UNVALIDATED_VERSION"


class UnvalidatedCalibVersionError(ValueError):
    """Raised when :func:`calib` is asked to apply constants for a detector
    whose full VERSION TRIPLE is not in the family's validated set (CAL-15).

    Refuse-by-default (mirrors :class:`pscalib.model.StaleConstantsError`):
    pscalib has a *single* apply plugin per family, and the version triple --
    which family-token dispatch discards -- is exactly what psana uses to pick
    the calib algorithm.  Rather than silently apply the one plugin to a version
    it was never byte-exactness-tested against, pscalib refuses and names the
    class, family, and the plugin it WOULD have used.  Carries ``det_class`` /
    ``family`` / ``plugin_name`` / ``validated`` for a caller to report or act
    on.  Subclasses :class:`ValueError` (it is an argument-value problem) so a
    caller catching ``ValueError`` gets the refusal, never a silent guess.
    """

    def __init__(self, det_class, family, plugin_name, validated):
        self.det_class = det_class
        self.family = family
        self.plugin_name = plugin_name
        self.validated = sorted(validated)
        super().__init__(
            f"pscalib refuses to calibrate {det_class!r}: this is an "
            f"UNVALIDATED version of the {family!r} detector family. pscalib "
            f"has a SINGLE {family!r} apply plugin ({plugin_name}), and this "
            f"repo's byte-exactness oracle has only validated it against "
            f"{self.validated!r}. psana chooses the calib ALGORITHM from the "
            f"full version triple (the deployed release ships more than one "
            f"epix10ka calib implementation), and family dispatch would discard "
            f"exactly that token -- so applying {plugin_name} to {det_class!r} "
            f"here would be a SILENT GUESS with no signal that it was never "
            f"validated. If you have independently confirmed {plugin_name} is "
            f"byte-exact for {det_class!r}, add it to "
            f"pscalib.registry._VALIDATED_VERSIONS[{family!r}], or pass "
            f"allow_unvalidated=True to calib(...) (or set env "
            f"{_ENV_ALLOW_UNVALIDATED}=1) to proceed at your own risk.")


def validated_versions(family=None):
    """The validated version triples (CAL-15).

    With ``family`` (a family token or a psana class name -- it is normalized),
    return the sorted list of validated psana class names for that family.
    Without it, return a ``{family: [class, ...]}`` snapshot of the whole map.
    """
    if family is None:
        return {f: sorted(v) for f, v in _VALIDATED_VERSIONS.items()}
    return sorted(_VALIDATED_VERSIONS.get(detector_type_of(family), ()))


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


def _version_identity(det_type):
    """Return the full versioned psana class name if ``det_type`` carries a
    version triple, else ``None`` (CAL-15).

    psana drp class names always carry the ``raw`` interface marker followed by
    a version triple (``epix10ka_raw_2_0_1``, ``jungfrau_raw_0_1_0``); the
    ``_raw_<x>_<y>_<z>`` separator is what distinguishes a specific *deployed
    class* (which names an algorithm psana would pick) from a bare family token
    or short detname (``epix10ka``, ``epixquad``, ``jungfrau``), which carries
    no version to validate.  ``detector_type_of`` throws the version away for
    LOOKUP; this keeps it for the validation gate.
    """
    if det_type is None:
        return None
    s = str(det_type)
    # a bare family / short detname (no interface marker) makes no version
    # claim -- there is nothing to validate, so dispatch as before.  Anchor on
    # the real psana ``_raw_`` separator (trailing underscore, exactly as
    # detector_type_of splits) so a detname merely CONTAINING "_raw" is not
    # misclassified as a versioned class name.
    return s if "_raw_" in s else None


def _env_allows_unvalidated():
    """True iff the CAL-15 env escape hatch is set to a truthy value."""
    return os.environ.get(_ENV_ALLOW_UNVALIDATED, "").strip().lower() in (
        "1", "true", "yes", "on")


def _assert_version_validated(det_type, family, allow_unvalidated=False):
    """CAL-15 gate: refuse to dispatch a version this family's single plugin was
    never validated against, unless the caller explicitly opts in.

    No-op when ``det_type`` carries no version (a bare family / short detname --
    :func:`_version_identity` returns ``None``) or the version is in the
    family's validated set.  Otherwise raises
    :class:`UnvalidatedCalibVersionError` (refuse-by-default) unless
    ``allow_unvalidated`` or the ``PSCALIB_ALLOW_UNVALIDATED_VERSION`` env flag
    is set.
    """
    version = _version_identity(det_type)
    if version is None:
        return
    # An UNREGISTERED family (no plugin at all) is an unsupported-DETECTOR
    # problem, not a version-validation one.  Skip the gate and let get_plugin
    # raise its clear "no apply plugin registered for ..." KeyError, rather than
    # a misleading refusal that names plugin 'None' and tells the caller to add
    # the version to _VALIDATED_VERSIONS.  Only families that actually have a
    # single plugin can be "guessing" about which version that plugin fits.
    if family not in _REGISTRY:
        return
    if version.lower() in _VALIDATED_VERSIONS.get(family, frozenset()):
        return
    if allow_unvalidated or _env_allows_unvalidated():
        return
    plugin = _REGISTRY.get(family)
    plugin_name = getattr(plugin, "__name__", repr(plugin))
    raise UnvalidatedCalibVersionError(
        version, family, plugin_name, _VALIDATED_VERSIONS.get(family, ()))


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
# COR-03: bind the constants to the raw BEFORE applying them
# ==========================================================================
# The bug (COR-03): calib(det_type, raw, constants) dispatches to the family
# plugin and applies ``constants`` to ``raw`` with NOTHING checking that the
# constants actually belong to this raw.  Same-family constants of the WRONG
# detector -- a different panel count, a different per-segment pixel geometry,
# or a different jungfrau entirely -- are applied silently: jungfrau raw +
# another jungfrau's pedestals/gain yields a finite ``(N,512,1024)`` image, no
# error, a large max-abs-diff.  A scientist can calibrate detector A's data with
# detector B's constants and publish a plausible-looking, wholly WRONG result.
#
# The apply leaves loop the raw's segments and index the constants per segment
# (``gfac[stage, s]`` for ``s in range(raw.shape[0])``), so a constants stack
# with MORE segments than the raw is silently tolerated (it indexes the leading
# segments) and a per-pixel geometry mismatch either broadcasts wrong or blows
# up deep in numpy with an opaque message.  The raw handed to :func:`calib` is a
# BARE ndarray -- it carries no detector id -- so the strongest correspondence
# bindable at this seam is the SHAPE the constants must project onto the raw:
# every per-segment spatial constant must cover exactly the raw's segment set
# and the raw's per-segment pixel geometry.  This is enforced here, BEFORE
# dispatch, so a non-corresponding set raises a clear
# :class:`ConstantsRawMismatchError` (naming raw shape vs constants shape)
# instead of silently producing a wrong image.
#
# RESIDUAL GAP (documented deliberately): two DIFFERENT detectors of the SAME
# family and SAME shape -- the literal COR-03 example, jungfrau A vs jungfrau B
# both ``(32,512,1024)`` -- are NOT separable by shape alone.  Catching that
# needs a detector IDENTITY on BOTH sides, but the raw is a bare ndarray with no
# id to compare the constants' ``detname``/``detector_uniqueid`` (CAL-07
# provenance) against.  Shape binding closes the wrong-panel-count /
# wrong-geometry class; same-shape-different-detector is only catchable once the
# raw itself carries (or the caller passes) its own identity.

#: The per-segment spatial constants the apply leaves actually consume.  Each is
#: shaped ``(..., n_segments, rows, cols)`` -- a gain-indexed 4-D constant
#: (``(G, S, rows, cols)``) or the 3-D ``mask`` (``(S, rows, cols)``) -- so its
#: segment count is ``shape[-3]`` and its per-segment geometry ``shape[-2:]``.
_SPATIAL_CTYPES = ("pedestals", "pixel_gain", "pixel_offset", "pixel_status",
                   "mask")


class ConstantsRawMismatchError(ValueError):
    """Raised when :func:`calib` is handed constants that do not correspond to
    the ``raw`` they would be applied to (COR-03).

    Nothing else binds the two: the apply plugin loops the raw's segments and
    indexes the constants per segment, so constants for a *different* detector
    of the same family (a different panel count or per-segment pixel geometry)
    would be applied silently and yield a finite but wholly WRONG image.  This
    refuses first, naming the offending ctype, the raw shape and the constants
    shape.  Subclasses :class:`ValueError` (an argument-value problem) so a
    caller catching ``ValueError`` gets the refusal, never a silent guess.
    Carries ``ctype`` / ``raw_shape`` / ``const_shape`` / ``family`` /
    ``raw_segments`` / ``const_segments`` for a caller to report or act on.
    """

    def __init__(self, ctype, raw_shape, const_shape, family,
                 raw_segments, const_segments):
        self.ctype = ctype
        self.raw_shape = tuple(int(d) for d in raw_shape)
        self.const_shape = tuple(int(d) for d in const_shape)
        self.family = family
        self.raw_segments = int(raw_segments)
        self.const_segments = int(const_segments)
        super().__init__(
            f"pscalib refuses to calibrate: the {family!r} constants do NOT "
            f"correspond to this raw. Constant {ctype!r} has shape "
            f"{self.const_shape} ({self.const_segments} segment(s) of "
            f"{self.const_shape[-2:]}), but raw has shape {self.raw_shape} "
            f"({self.raw_segments} segment(s) of {self.raw_shape[-2:]}). "
            f"Nothing binds constants to the raw they are applied to, so "
            f"applying a DIFFERENT detector's constants (wrong panel count / "
            f"pixel geometry) would silently produce a finite but WRONG "
            f"calibrated image. Pass constants captured for THIS exact "
            f"detector+run (a snapshot or web fetch pinned to it).")


def _assert_constants_bind_raw(family, raw, constants):
    """COR-03 gate: refuse to apply constants whose per-segment spatial shape
    does not correspond to ``raw`` (segment count + per-segment pixel geometry).

    No-op unless ``raw`` is a 3-D ``(n_segments, rows, cols)`` stack (the shape
    every registered plugin consumes); a non-3-D raw is left for the plugin's
    own ``raw must be 3-D`` error.  For each per-segment spatial constant that is
    actually present (:data:`_SPATIAL_CTYPES`), require its segment count
    (``shape[-3]``) and per-segment geometry (``shape[-2:]``) to equal the
    raw's; a mismatch raises :class:`ConstantsRawMismatchError`.  Absent or
    non-spatial (rank < 3) ctypes are skipped -- a missing REQUIRED ctype is the
    plugin's own clear ``KeyError``, not a correspondence problem, and a
    correctly-corresponding set (every existing byte-exact oracle) passes
    untouched so the calibrated output is byte-unchanged.
    """
    arr = np.asarray(raw)
    if arr.ndim != 3:
        return                          # let the plugin raise its own ndim error
    raw_segs, raw_rows, raw_cols = (int(d) for d in arr.shape)
    for ctype in _SPATIAL_CTYPES:
        c = _get_const(constants, ctype, required=False)
        if c is None:
            continue
        c = np.asarray(c)
        # a per-segment spatial constant is (..., n_segments, rows, cols); it
        # needs at least those three trailing axes to name a segment + geometry.
        if c.ndim < 3:
            continue
        const_segs = int(c.shape[-3])
        const_rows, const_cols = int(c.shape[-2]), int(c.shape[-1])
        if (const_rows, const_cols) != (raw_rows, raw_cols) \
                or const_segs != raw_segs:
            raise ConstantsRawMismatchError(
                ctype, arr.shape, c.shape, family, raw_segs, const_segs)


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
    # An EMPTY config (e.g. seg_configs returned {} because the per-segment
    # CONFIGURE-block could not be located) would otherwise fail deep in the
    # gain decode with an opaque ``np.stack([])`` ("need at least one array to
    # stack").  Fail here with a diagnosable message instead.
    if hasattr(config, "__len__") and len(config) == 0:
        raise ValueError(
            "epix10ka apply got an EMPTY per-segment config "
            "(config has no segments).  psdata's run.seg_configs(detname) "
            "returned nothing -- the detector's CONFIGURE-block (trbit / "
            "asicPixelConfig) was not found in any front transition dgram, so "
            "the gain-range decode cannot run.")
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


def calib(*args, config=None, run=None, allow_stale=False, log=None,
          allow_unvalidated=False):
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

    Version-dispatch guard (CAL-15) is wired in: dispatch normalizes to the
    detector *family* (one plugin per family), but psana chooses the calib
    *algorithm* from the full version triple.  When the detector class carries a
    version triple that this family's single plugin was never byte-exactness
    validated against, ``calib`` REFUSES (raises
    :class:`UnvalidatedCalibVersionError`) rather than silently guessing -- pass
    ``allow_unvalidated=True`` (or set env
    ``PSCALIB_ALLOW_UNVALIDATED_VERSION=1``) to override.  A bare family token /
    short detname (no version) or a validated version dispatches as before.

    Constants<->raw binding (COR-03) is wired in: nothing else checks that the
    ``constants`` actually correspond to the ``raw`` they are applied to, so a
    different detector's same-family constants (wrong panel count / per-segment
    pixel geometry) would be applied silently and yield a finite but WRONG
    image.  Before dispatch, ``calib`` verifies every per-segment spatial
    constant covers exactly the raw's segment set and per-segment geometry, and
    raises :class:`ConstantsRawMismatchError` (naming raw shape vs constants
    shape) on a mismatch.  A correctly-corresponding set is byte-unchanged.  The
    residual gap: two DIFFERENT detectors of the SAME family and SAME shape are
    not separable by shape alone (the raw is a bare ndarray with no identity to
    bind the constants' provenance against).

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
    allow_unvalidated : bool
        Opt in to applying the family plugin to a version triple that is not in
        the family's validated set (CAL-15).  Refuse-by-default (``False``);
        set ``True`` only when you have independently confirmed byte-exactness.

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
        # CAL-15: the caller named a specific class -- keep its full version and
        # gate on it (refuse-by-default for an unvalidated version triple).
        version_hint = det_type
    else:
        # inferred form: calib(raw, constants)
        if len(args) != 2:
            raise TypeError(
                "calib(raw, constants, config=..., run=...) takes raw and "
                f"constants; got {len(args)} positional args (for an explicit "
                "detector type use calib(det_type, raw, constants, ...))")
        raw, constants = args
        norm = detector_type_for_constants(constants)
        # CAL-15: recover the FULL identity the constants carry (a snapshot's
        # detname, a web fetch's dettype, ...) so a version triple is validated
        # here too, not silently normalized away.  (Real snapshots carry only a
        # bare family/short detname, so this is a no-op for them -- but if a
        # provider ever names a specific class it is gated, not guessed.)
        from .model import detector_type_hint
        version_hint = detector_type_hint(constants)

    _assert_version_validated(version_hint, norm, allow_unvalidated)
    # COR-03: bind the constants to the raw (shape/segment correspondence)
    # before dispatch, so a different detector's same-family constants raise a
    # clear error instead of silently producing a wrong image.
    _assert_constants_bind_raw(norm, raw, constants)
    _enforce_validity(constants, run, allow_stale, log)
    return get_plugin(norm)(raw, constants, config=config)
