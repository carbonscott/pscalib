"""pscalib.model -- the constants contract: ``Pin``, ``Validity``, enforcement.

This module is the single home of pscalib's *validity / staleness* model.  It is
pure-python + numpy-free (it carries no arrays, only the metadata that pins a set
of constants to a detector + run and the run-range each constant is valid for).

The one correctness feature pscalib adds beyond the lifted psdata prototype is
**refuse-by-default staleness enforcement** (US-002):

  * Every constant carries a :class:`Validity` -- the ``[run, run_end]`` range it
    is valid for, parsed from its metadata doc exactly as psana's
    ``CalibDoc`` / ``select_doc_in_run_range`` (``MDBWebUtils.py:196``) parse it:
    ``run`` is the first valid run, ``run_end`` the last, with the sentinel
    string ``'end'`` meaning open-ended (capped at :data:`Validity.RUN_MAX`).

  * Applying constants to raw from a run *outside* every constant's range
    :func:`raises <check_validity>` :class:`StaleConstantsError` **by default**.
    ``allow_stale=True`` downgrades the refusal to a logged warning; an in-range
    apply passes silently.

psdata's ``snapshot.py`` ``is_valid_for_run`` is advisory only (returns a bool,
never refuses); this module turns that into enforcement.  The selection rule
itself (which doc you *get* for a run) lives in
:func:`pscalib.providers.webdb.select_doc_in_run_range`; this module enforces that
the constants you *already hold* still cover the run you are calibrating.
"""

import logging

__all__ = [
    "StaleConstantsError",
    "Validity",
    "Pin",
    "Constants",
    "validity_from_meta",
    "validities_from_calibconst",
    "detector_type_hint",
    "check_validity",
]

logger = logging.getLogger(__name__)


class StaleConstantsError(Exception):
    """Raised when calibration constants are applied to a run *outside* their
    validity range and staleness was not explicitly allowed.

    Refuse-by-default: see :func:`check_validity`.  Carries the offending
    ``run`` and the list of ``(ctype, Validity)`` pairs that do not cover it, so
    a caller can report exactly which constants are stale.
    """

    def __init__(self, run, offenders, pin=None):
        self.run = int(run)
        #: list of ``(ctype, Validity)`` whose range does not cover ``run``.
        self.offenders = list(offenders)
        #: the :class:`Pin` the constants were taken for, if known.
        self.pin = pin
        detail = ", ".join(f"{ct}={v}" for ct, v in self.offenders)
        msg = (f"calibration constants are STALE for run {self.run}: "
               f"{len(self.offenders)} ctype(s) out of range [{detail}]")
        if pin is not None:
            msg += f"; constants pinned at {pin}"
        msg += (" -- pass allow_stale=True to apply anyway (downgrades to a "
                "warning)")
        super().__init__(msg)


class Validity:
    """The run-range a single calibration constant is valid for.

    Mirrors psana's ``CalibDoc`` (``CalibDoc.py``) parse of a metadata doc:
    ``run`` is the first valid run number, ``run_end`` the last.  The sentinel
    string ``'end'`` (or ``None``) means *open-ended* and is represented as
    :data:`RUN_MAX` (psana's ``CalibDoc.rnum_max == 9999``).

    A :class:`Validity` is immutable, hashable, and cheaply comparable.

    Attributes
    ----------
    run : int
        First run the constant is valid for (psana ``CalibDoc.begin``).
    run_end : int
        Last run the constant is valid for (psana ``CalibDoc.end``); equals
        :data:`RUN_MAX` for the open-ended ``'end'`` sentinel.
    open_ended : bool
        True iff the source metadata used the ``'end'`` sentinel (i.e. the range
        extends to :data:`RUN_MAX`).
    """

    #: psana ``CalibDoc.rnum_max`` -- the cap an open-ended (``'end'``) range maps
    #: to, and the maximum legal run number for a validity bound.
    RUN_MAX = 9999

    __slots__ = ("run", "run_end", "open_ended")

    def __init__(self, run, run_end="end"):
        run = int(run)
        if run < 0 or run > self.RUN_MAX:
            raise ValueError(
                f"validity 'run' must be in [0, {self.RUN_MAX}]; got {run}")
        self.run = run

        if run_end is None or (isinstance(run_end, str)
                               and run_end.lower() == "end"):
            self.run_end = self.RUN_MAX
            self.open_ended = True
        else:
            # accept int or a digit string (psana stores run_end as either)
            if isinstance(run_end, str):
                if not run_end.isdigit():
                    raise ValueError(
                        f"invalid validity 'run_end' value {run_end!r} "
                        f"(expected an int, a digit string, or 'end')")
                run_end = int(run_end)
            run_end = int(run_end)
            if run_end > self.RUN_MAX:
                raise ValueError(
                    f"validity 'run_end' {run_end} exceeds RUN_MAX "
                    f"{self.RUN_MAX}")
            if run_end < self.run:
                raise ValueError(
                    f"validity 'run_end' {run_end} precedes 'run' {self.run}")
            self.run_end = run_end
            self.open_ended = False

    @classmethod
    def from_meta(cls, meta):
        """Build a :class:`Validity` from a constant's metadata doc.

        ``meta`` is the per-ctype metadata dict -- psana's ``det.raw._calibconst``
        attaches it as the second element of each ``(data, meta)`` pair, and the
        snapshot manifest keeps it under ``validity[ctype]``.  Reads the ``run``
        and ``run_end`` fields (the same two ``CalibDoc`` reads).

        Raises ``KeyError`` if ``run`` is absent (a constant with no first-valid
        run is not a parseable validity range).
        """
        if not isinstance(meta, dict):
            raise TypeError(
                f"validity metadata must be a dict; got {type(meta).__name__}")
        if "run" not in meta or meta["run"] is None:
            raise KeyError(
                "validity metadata has no 'run' (first-valid-run) field")
        return cls(meta["run"], meta.get("run_end", "end"))

    def contains(self, run):
        """True iff ``run`` falls within ``[run, run_end]`` (inclusive).

        Same test as psana ``select_doc_in_run_range``: ``begin <= rnum <= end``.
        """
        run = int(run)
        return self.run <= run <= self.run_end

    def as_dict(self):
        """Return ``{'run', 'run_end'}`` with ``run_end`` re-encoded as the
        ``'end'`` sentinel when open-ended (round-trips the source metadata)."""
        return {"run": self.run,
                "run_end": "end" if self.open_ended else self.run_end}

    def __eq__(self, other):
        return (isinstance(other, Validity)
                and self.run == other.run
                and self.run_end == other.run_end)

    def __hash__(self):
        return hash((self.run, self.run_end))

    def __repr__(self):
        end = "'end'" if self.open_ended else self.run_end
        return f"Validity(run={self.run}, run_end={end})"


class Pin:
    """The ``(detector_uniqueid, run)`` identity a set of constants is pinned to.

    A snapshot or a web fetch is taken *for* a specific detector and run; the
    :class:`Pin` records that provenance.  ``run`` here is "the run you asked to
    calibrate" (the snapshot/fetch run), not any single constant's first-valid
    run -- the latter lives in each constant's :class:`Validity`.

    Attributes
    ----------
    detector_uniqueid : str
        ``det.raw._uniqueid`` at capture time -- the long unique id used as the
        DB query key.
    run : int
        The run the constants were captured for.
    detname : str | None
        Detector short name (e.g. ``"jungfrau"``), if known.
    exp : str | None
        Experiment id (e.g. ``"mfx100848724"``), if known.
    """

    __slots__ = ("detector_uniqueid", "run", "detname", "exp")

    def __init__(self, detector_uniqueid, run, detname=None, exp=None):
        self.detector_uniqueid = detector_uniqueid
        self.run = int(run)
        self.detname = detname
        self.exp = exp

    @classmethod
    def from_snapshot_pin(cls, pin):
        """Build from a snapshot manifest's ``pin`` dict
        (``CalibSnapshot.pin``)."""
        return cls(detector_uniqueid=pin["detector_uniqueid"],
                   run=pin["run"],
                   detname=pin.get("detname"),
                   exp=pin.get("exp"))

    def as_dict(self):
        return {"detector_uniqueid": self.detector_uniqueid,
                "run": self.run,
                "detname": self.detname,
                "exp": self.exp}

    def __eq__(self, other):
        return (isinstance(other, Pin)
                and self.detector_uniqueid == other.detector_uniqueid
                and self.run == other.run)

    def __hash__(self):
        return hash((self.detector_uniqueid, self.run))

    def __repr__(self):
        d = f", detname={self.detname!r}" if self.detname else ""
        e = f", exp={self.exp!r}" if self.exp else ""
        return (f"Pin(detector_uniqueid={self.detector_uniqueid!r}, "
                f"run={self.run}{d}{e})")


# ==========================================================================
# Helpers: extract per-ctype validity from a metadata source
# ==========================================================================
def validity_from_meta(meta):
    """Parse one constant's metadata doc into a :class:`Validity` (alias of
    :meth:`Validity.from_meta`)."""
    return Validity.from_meta(meta)


def validities_from_calibconst(calibconst):
    """Map a ``{ctype: (data, meta)}`` calibconst dict (psana ``_calibconst``,
    ``CalibSnapshot.calibconst()``, or ``webdb.get_constants()``) to
    ``{ctype: Validity}``.

    ctypes whose metadata has no parseable ``run`` field are skipped (they carry
    no enforceable range) rather than raising -- enforcement is over the ctypes
    that *do* declare a range.
    """
    out = {}
    for ctype, value in calibconst.items():
        meta = value[1] if isinstance(value, (tuple, list)) and len(value) > 1 \
            else None
        if not isinstance(meta, dict):
            continue
        try:
            out[ctype] = Validity.from_meta(meta)
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ==========================================================================
# Detector-type hint extraction (US-005: infer the apply plugin from the
# constants alone, so the public surface needs no det_type argument)
# ==========================================================================
#: Metadata keys (in priority order) that name a constant's detector type/name.
#: psana attaches ``dettype`` (the bare family, e.g. ``'epix10ka'``) and
#: ``detname`` (the short name, e.g. ``'epixquad'``) to every ``_calibconst``
#: doc; either resolves to the right plugin once normalized.  See
#: :data:`pscalib.providers.snapshot._META_KEEP`.
_DETTYPE_META_KEYS = ("dettype", "detname", "detector")


def detector_type_hint(constants):
    """Best-effort detector-type/name string carried *by the constants*.

    The US-005 public surface (:func:`pscalib.calib`) infers which apply plugin
    to dispatch to from the constants alone -- it takes no ``det_type``
    argument.  This pulls that hint, trying, in order:

    * a :class:`pscalib.providers.snapshot.CalibSnapshot` -- its ``detname`` /
      per-ctype ``dettype`` metadata (and the ``detector_uniqueid`` prefix);
    * a :class:`Constants` adapter (delegates to its wrapped source);
    * a psana-style ``{ctype: (data, meta)}`` dict -- the ``dettype`` /
      ``detname`` / ``detector`` field of any ctype's metadata doc;
    * an explicit ``det_type`` / ``dettype`` / ``detname`` key on a plain dict.

    Returns the raw hint string (e.g. ``'epix10ka'``, ``'epixquad'``,
    ``'jungfrau'``) for :func:`pscalib.registry.detector_type_of` to normalize,
    or ``None`` if the constants carry no recoverable detector identity.
    """
    if constants is None:
        return None

    # a Constants adapter knows its own source -- unwrap to the real thing
    if isinstance(constants, Constants):
        constants = constants.source

    # a CalibSnapshot exposes detname directly + dettype in per-ctype metadata
    detname = getattr(constants, "detname", None)
    if detname:
        return detname
    uid = getattr(constants, "detector_uniqueid", None)
    if isinstance(uid, str) and "_" in uid:
        # uniqueid looks like '<family>_<serial>...' (e.g. 'epix10ka_...')
        return uid.split("_", 1)[0]

    # mapping forms: scan ctype metadata docs, then explicit naming keys
    if hasattr(constants, "items"):
        for _ctype, value in constants.items():
            meta = value[1] if isinstance(value, (tuple, list)) \
                and len(value) > 1 else None
            if isinstance(meta, dict):
                for k in _DETTYPE_META_KEYS:
                    if meta.get(k):
                        return meta[k]
        for k in ("det_type",) + _DETTYPE_META_KEYS:
            if constants.get(k):
                return constants[k]
    return None


# ==========================================================================
# The uniform Constants contract (US-005)
# ==========================================================================
class Constants:
    """A uniform, provider-agnostic view over a set of calibration constants.

    US-005's "one uniform :class:`Constants` contract": whatever provider the
    constants came from -- a snapshot (US-000), a web fetch (US-001), or a
    caller-supplied (BYO) dict -- the apply path sees the same small surface:

      * :meth:`array` -- the ndarray for a ctype (or ``None``);
      * :meth:`validities` -- ``{ctype: Validity}`` for staleness enforcement;
      * :attr:`det_type_hint` -- the detector-type/name the constants name
        themselves with (so :func:`pscalib.calib` needs no ``det_type`` arg);
      * :attr:`source` -- the wrapped object, passed *unchanged* to the plugin.

    This is a thin, numpy-free *adapter*, not a copy: it holds a reference to the
    wrapped source and forwards lookups.  The registry's apply plugins still
    accept the bare source directly (a dict / snapshot), so ``Constants`` is
    optional sugar that makes the contract explicit and testable -- wrapping is
    idempotent (``Constants(Constants(x)).source is x``).

    Parameters
    ----------
    source : Mapping | CalibSnapshot
        A plain ``{ctype: ndarray}`` dict, a psana-style ``{ctype: (ndarray,
        meta)}`` dict, or a :class:`pscalib.providers.snapshot.CalibSnapshot`.
    pin : Pin | None
        The ``(detector_uniqueid, run)`` identity, if known (a snapshot/web
        fetch carries one; a BYO dict may not).
    """

    __slots__ = ("source", "_pin")

    def __init__(self, source, pin=None):
        if source is None:
            raise ValueError("Constants source must not be None")
        # idempotent: wrapping a Constants returns a view on the same source
        if isinstance(source, Constants):
            pin = pin if pin is not None else source._pin
            source = source.source
        self.source = source
        self._pin = pin

    @classmethod
    def of(cls, source, pin=None):
        """Coerce ``source`` to a :class:`Constants` (idempotent).  ``Constants``
        instances are returned as-is; everything else is wrapped."""
        if isinstance(source, cls):
            return source
        return cls(source, pin=pin)

    def array(self, ctype):
        """Return the ndarray for ``ctype`` (``None`` if absent), unwrapping the
        psana-style ``(ndarray, meta)`` tuple form when present."""
        src = self.source
        if hasattr(src, "array") and callable(src.array):
            if ctype == "mask":
                m = getattr(src, "mask", None)
                return m if m is not None else src.array("mask")
            return src.array(ctype)
        val = None
        if hasattr(src, "get"):
            val = src.get(ctype)
        else:
            try:
                val = src[ctype]
            except (KeyError, TypeError, IndexError):
                val = None
        if isinstance(val, (tuple, list)) and val and not isinstance(val, str) \
                and hasattr(val[0], "shape"):
            val = val[0]
        return val

    def calibconst(self):
        """Return the underlying ``{ctype: (data, meta)}`` calibconst mapping,
        reconstructing it from a :class:`CalibSnapshot` when needed."""
        src = self.source
        if hasattr(src, "calibconst") and callable(src.calibconst):
            return src.calibconst()
        return dict(src) if hasattr(src, "items") else {}

    def validities(self):
        """``{ctype: Validity}`` for the wrapped constants (US-002 enforcement
        input).  Empty if the constants carry no parseable validity metadata
        (e.g. a bare BYO ``{ctype: ndarray}`` dict)."""
        src = self.source
        if hasattr(src, "validities") and callable(src.validities):
            return src.validities()
        return validities_from_calibconst(self.calibconst())

    @property
    def pin(self):
        """The :class:`Pin` the constants were taken for, or ``None``.  Falls
        back to a wrapped snapshot's ``pin_obj``."""
        if self._pin is not None:
            return self._pin
        po = getattr(self.source, "pin_obj", None)
        return po

    @property
    def det_type_hint(self):
        """The detector-type/name string the constants name themselves with
        (for :func:`pscalib.registry.detector_type_of` to normalize), or
        ``None``."""
        return detector_type_hint(self.source)

    def __repr__(self):
        return (f"Constants(source={type(self.source).__name__}, "
                f"det_type_hint={self.det_type_hint!r}, pin={self.pin})")


# ==========================================================================
# CAL-05: intersect the staleness check with the apply plugin's READ SET
# ==========================================================================
# The bug (CAL-05): the staleness guard checked the validity range of *every*
# ctype the constants carry and refused if ANY was out of range -- including
# ctypes the detector's apply plugin PROVABLY NEVER READS.  On the demo run
# (ued1010667/r177, an epix10ka-family detector) the ``geometry`` doc is out of
# validity range for the run, so the guard raised ``StaleConstantsError`` even
# though ``geometry`` never touches the calibrated output: the epix10ka apply
# reads only ``pedestals`` / ``pixel_gain`` / ``pixel_status`` / ``mask`` (see
# ``pscalib.apply.epix10ka`` + ``registry.plugin_epix10ka``), and the assembled
# image uses PRE-CACHED pixel-index maps, not the live geometry doc.  psana
# returns a finite array; pscalib's false refusal made it strictly worse.
#
# The fix: enforce staleness ONLY over the ctypes the family's apply plugin
# actually consumes -- ``present_ctypes ∩ read_set``.  A stale ctype the plugin
# never reads (``geometry`` for epix10ka) no longer causes a refusal; a stale
# ctype the plugin DOES read still fires the guard (it is NOT disabled, only
# narrowed).
#
# The read-sets are DATA, seeded from exactly what each apply plugin pulls
# (``registry._get_const(constants, <ctype>)`` in ``plugin_epix10ka`` /
# ``plugin_jungfrau``, and the ``calib_*`` leaves they call):
#   * epix10ka -- pedestals, pixel_gain, pixel_status, mask
#                 (NOT pixel_offset, NOT geometry)
#   * jungfrau -- pedestals, pixel_gain, pixel_offset, pixel_status, mask
#                 (NOT geometry)
# ``pixel_status`` is included because both plugins derive the default mask from
# it (``mask_from_pixel_status``) when no cached ``mask`` is present.
#: family token -> frozenset of the ctypes that family's apply plugin reads.
_PLUGIN_READS = {
    "epix10ka": frozenset({"pedestals", "pixel_gain", "pixel_status", "mask"}),
    "jungfrau": frozenset({"pedestals", "pixel_gain", "pixel_offset",
                           "pixel_status", "mask"}),
}


def _normalize_family(name):
    """Normalize a detector type/class/detname to its family token, or ``None``.

    A numpy-free mirror of :func:`pscalib.registry.detector_type_of` (kept local
    so :mod:`pscalib.model` stays free of the registry -- and of the numpy the
    registry pulls in): strip a psana ``<family>_raw_x_y_z`` suffix, lowercase,
    and fold the epix10ka composite/family aliases onto ``"epix10ka"``.
    """
    if name is None:
        return None
    s = str(name)
    for sep in ("_raw_", "_raw"):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    s = s.lower()
    if s.startswith("epix10ka") or s in ("epixquad", "epix10kaquad",
                                         "epix10ka2m"):
        return "epix10ka"
    return s


def _family_from_pin(pin):
    """Best-effort detector family carried by a :class:`Pin`, or ``None``.

    The :class:`Pin` that already flows into :func:`check_validity` (a snapshot's
    ``pin_obj`` or a web fetch's pin) names its detector -- ``detname``
    (``"epixquad"`` / ``"jungfrau"``) or the ``detector_uniqueid`` family prefix
    (``"epix10ka_..."``).  This is how the family reaches the guard WITHOUT
    changing any caller: the read-set is recovered from the pin already passed.
    """
    if pin is None:
        return None
    detname = getattr(pin, "detname", None)
    if detname:
        return detname
    uid = getattr(pin, "detector_uniqueid", None)
    if isinstance(uid, str) and "_" in uid:
        return uid.split("_", 1)[0]
    return None


def _read_set_for(family, pin):
    """The ctype read-set to enforce staleness over, or ``None`` for "check all".

    Resolves the family from an explicit ``family`` argument first, else from the
    ``pin`` already passed to :func:`check_validity`.  Returns the family's
    :data:`_PLUGIN_READS` set when it is a known family, or ``None`` when the
    family cannot be determined or is not a registered leaf -- in which case the
    guard falls back to checking EVERY ctype (the pre-CAL-05 behavior), so the
    change is backward-compatible and never widens a refusal beyond what a known
    plugin reads.
    """
    fam = _normalize_family(family if family is not None
                            else _family_from_pin(pin))
    if fam is not None and fam in _PLUGIN_READS:
        return _PLUGIN_READS[fam]
    return None


# ==========================================================================
# THE enforcement entry point (US-002)
# ==========================================================================
def check_validity(validities, run, allow_stale=False, pin=None, log=None,
                   family=None):
    """Enforce that constants are valid for ``run`` -- refuse-by-default.

    This is the one correctness feature pscalib adds over psdata's advisory
    ``is_valid_for_run``.  Given a ``{ctype: Validity}`` map (from
    :func:`validities_from_calibconst`) and the run being calibrated:

    * **in range** (every *enforced* ctype's :meth:`Validity.contains` is True)
      -- returns silently.
    * **out of range** and ``allow_stale=False`` (the default) -- raises
      :class:`StaleConstantsError` naming every offending ctype.
    * **out of range** and ``allow_stale=True`` -- logs a single ``warning`` and
      returns (the apply proceeds with stale constants).

    CAL-05 -- staleness is enforced only over the ctypes the detector family's
    apply plugin actually READS (``present_ctypes ∩ read_set``; see
    :data:`_PLUGIN_READS`).  A stale ctype the plugin never consumes (e.g.
    ``geometry`` for epix10ka -- the calibrated output and the pre-cached image
    grid never touch it) does NOT trigger a refusal, matching psana, which
    returns a finite array.  The read-set is recovered from ``family`` if given,
    else from the ``pin`` already passed (its ``detname`` / ``detector_uniqueid``
    names the detector) -- so the narrowing needs no change at the call site.
    When the family cannot be determined (no ``family`` and no identifying
    ``pin``) or is not a known plugin, EVERY ctype is checked (the pre-CAL-05
    behavior), so the change is backward-compatible and never disables the guard.

    Parameters
    ----------
    validities : dict
        ``{ctype: Validity}`` -- the per-ctype ranges to check.
    run : int
        The run whose raw data the constants are about to calibrate.
    allow_stale : bool
        If True, downgrade an out-of-range refusal to a logged warning.
    pin : Pin | None
        The pin the constants were taken for (for the error/warning message,
        and -- CAL-05 -- as the fallback source of the detector family whose
        read-set the check is intersected with).
    log : logging.Logger | None
        Logger to warn on (defaults to this module's logger).
    family : str | None
        Detector type / family / class name (CAL-05).  When given (or recovered
        from ``pin``) and it maps to a known apply plugin, staleness is enforced
        only over that plugin's read-set (:data:`_PLUGIN_READS`).  ``None`` and
        an unknown family fall back to checking every ctype.

    Returns
    -------
    list of (ctype, Validity)
        The offenders (empty when in range).  When ``allow_stale`` is False this
        is always empty on return (it raised otherwise).

    Raises
    ------
    StaleConstantsError
        If out of range and ``allow_stale`` is False.
    """
    run = int(run)
    # CAL-05: enforce only over the ctypes the family's apply plugin reads.  A
    # read-set of None (unknown family) means "check all" (pre-CAL-05 behavior).
    read_set = _read_set_for(family, pin)
    items = validities.items()
    if read_set is not None:
        items = [(ct, v) for ct, v in items if ct in read_set]
    offenders = [(ct, v) for ct, v in sorted(items)
                 if not v.contains(run)]
    if not offenders:
        return []
    if not allow_stale:
        raise StaleConstantsError(run, offenders, pin=pin)
    log = log or logger
    detail = ", ".join(f"{ct}={v}" for ct, v in offenders)
    pin_s = f" (pinned at {pin})" if pin is not None else ""
    log.warning(
        "applying STALE calibration constants to run %d%s: %d ctype(s) out of "
        "range [%s] -- proceeding because allow_stale=True",
        run, pin_s, len(offenders), detail)
    return offenders
