"""pscalib.apply._fastcalib -- the byte-exact fast machinery behind the apply leaves.

This module holds THREE things, none of which changes a single output bit:

1. **The memo (the hoist).**  ``poff = pedestals + pixel_offset``,
   ``gfac = 1/pixel_gain`` and the default status mask are pure functions of the
   *calibration constants*, which are fetched once per run -- yet the c5ce538
   apply path rebuilt all three on EVERY event.  ``mask_from_pixel_status``
   alone was 45-56% of the whole per-event calib cost.  :func:`memo` caches them
   keyed on the IDENTITY of the source arrays.

   Keying on a bare ``id()`` is UNSOUND: CPython recycles addresses, so a freed
   array's id can be handed to a different array and a bare-id cache would
   silently return the wrong constants.  We therefore store a ``weakref.ref`` to
   each source next to the derived value, with a callback that POPS THIS EXACT
   KEY when the source dies (so the entry cannot outlive the id), plus a
   belt-and-braces ``ref() is src`` re-check on every hit.  Strong references are
   held ONLY to derived arrays, never to sources.  (``WeakKeyDictionary`` is not
   an option: ``ndarray.__hash__`` is ``None``.)

2. **The pure-numpy hybrid kernel.**  Per spatial tile, one of three strategies
   is chosen from cheap integer reductions OF THAT TILE.  The classifier is ONE
   bitwise-OR reduction; ``hi = OR(tile) & ~0x3fff``:

     A. ``hi == 0`` -- no pixel has any bit above bit 13, so every gain code is 0
        AND ``raw & 0x3fff == raw``: the whole tile is stage 0 and the ``& MSK``
        can be DROPPED.  One constant plane, one streaming pass.
     B. many non-G0 pixels -- a DENSE two-plane blend: compute the stage-0 AND
        the stage-1 result for EVERY pixel and ``np.copyto(..., where=)`` between
        the finished values.  Selection selects, it does not compute, so this is
        bit-exact; and nothing in it is proportional to how many pixels switched,
        which is what caps the fixup slope on high-non-G0 frames.
     C. few non-G0 pixels -- the stage-0 pass, then a SPARSE gather fixup that
        overwrites just those pixels.

   B-vs-C is decided per tile by the tile's own non-G0 count against
   :data:`DENSE_FRAC` (measured, not guessed).  Either way stage 2 (``gbits==3``)
   and the BAD code (``gbits==2``) -- together well under 0.002% of pixels -- are
   gathered.  Note ``a.max()`` would NOT be a sound classifier where OR is: a
   negative word of a signed raw dtype has a small max yet a nonzero truncated
   gain code, and a word with bit 22 set truncates to code 0 yet is not equal to
   ``raw & MSK``.

   Both branches evaluate, for every pixel, LITERALLY the reference expression
   ``((raw & MSK).astype(f32) - poff[stage]) * gfac[stage] * mask.astype(f32)``
   with the reference's operation order and its THREE separate float32
   roundings.  The path choice depends only on the frame's own contents, so it is
   invariant under any event partition; blocking is over SPATIAL axes only.

3. **The OPTIONAL fused numba backend.**  ONE flat float32 pass that decodes the
   gain code per pixel and touches only the plane that pixel needs, so the
   constant-plane traffic follows the SPATIAL CLUSTERING of the gain switching
   rather than the switch count.  Deliberately NOT block-dispatched like (2):
   both block-dispatched shapes were built and measured and both were slower
   (see :func:`_fused_seg`).  numba is imported behind ``try/except``: ``import
   pscalib`` works with numpy ALONE and falls back to (2).

Traps that are deliberately respected here (each one has already bitten):

  * ``-0.0`` IS LOAD-BEARING.  A masked pixel whose ``(adc - poff) * gfac`` is
    finite-negative becomes ``x * 0.0 == -0.0``.  Writing a literal ``0.0`` for
    masked pixels is BIT-WRONG.  Nothing here ever short-circuits the mask
    multiply.
  * NO DOUBLE ROUNDING.  The reference rounds THREE times (subtract, multiply,
    mask-multiply).  Computing the chain in float64 and storing once gives ONE
    rounding and differs on millions of pixels per frame.  Every intermediate
    here is float32, and in the numba kernel every literal is written
    ``np.float32(...)`` because a bare python float promotes to float64.
  * THE BAD CODE'S SIGN.  ``gbits == 2`` has no gain stage; the reference's
    ``np.select`` default gives it ``pedoff=0``/``factor=0``, i.e. it *computes*
    ``(adc - 0.0) * 0.0``, and that multiply decides the sign of the zero (and
    yields NaN for a non-finite mask).  We compute it the same way.
  * NaN PAYLOADS.  On x86 ``MULSS`` returns the DESTINATION operand when it is
    NaN, so ``m * v`` instead of ``v * m`` can change which payload propagates.
    Operand order is identical to the reference in every multiply.
  * SINGLE-THREADED.  ``parallel=False``, no ``prange``, no threads.  A hidden
    thread would manufacture a fake speedup at workers=1.

The reference this is written against is pscalib c5ce538,
``pscalib.apply.jungfrau.calib_jungfrau`` (preserved verbatim as
:func:`pscalib.apply.jungfrau.calib_jungfrau_reference`) and
``pscalib.apply.epix10ka.mask_from_pixel_status``.
"""

import os
import weakref

import numpy as np

__all__ = [
    "MSK", "BSH", "N_GAIN_STAGES",
    "memo", "memo_clear", "memo_stats", "memo_size",
    "derive_poff", "derive_gfac",
    "calib_jungfrau_fast", "backend_info",
    "NUMBA_AVAILABLE", "NUMBA_IMPORT_ERROR",
    "assert_kernels_are_float32_only",
]

#: 14-bit ADC mask -- psana ``UtilsJungfrau.MSK``.
MSK = 0x3fff
#: Gain-bit shift -- psana ``UtilsJungfrau.BSH``.
BSH = 14
#: Gain stages on the leading constant axis.
N_GAIN_STAGES = 3


def _env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


#: Rows per spatial tile for the numpy path.  128 rows x 1024 cols x 4 B = 512 KB
#: of float32, so a tile plus its scratch stays resident in L2/L3 while the
#: in-place ops run over it.  MEASURED best of {0(whole),512,256,128,64,32} on the
#: campaign fixture at every non-G0 fraction from 0 to 1 (job 34302162).
TILE_ROWS = _env_int("PSCALIB_CALIB_TILE_ROWS", 128)

#: Per-tile non-G0 FRACTION above which the dense two-plane blend beats the
#: sparse gather.  MEASURED, not guessed: forcing dense-on-every-tile against
#: gather-on-every-tile over a synthetic frac sweep 0 -> 1 (job 34302162, node
#: sdfmilan057, ms/frame at tile_rows=128) crosses over at frac ~ 0.25 --
#:   frac   0.002  0.010  0.030  0.080  0.150  0.250  0.400  0.600  1.000
#:   dense  30.6   35.4   40.8   51.4   58.6   61.9   67.4   74.9   38.1
#:   gather 26.4   33.3   36.8   45.3   56.9   71.0   93.0  119.5  175.3
#: The gather is cheaper below the crossover and its cost grows without bound
#: above it; the dense blend is flat-ish.  Note this was measured as a
#: WHOLE-FRAME fraction while the threshold is applied PER TILE, so it is an
#: estimate of the per-tile crossover, not a proof of it.
DENSE_FRAC = _env_float("PSCALIB_CALIB_DENSE_FRAC", 0.25)

#: ``auto`` (numba if importable, else numpy) | ``numpy`` | ``numba`` |
#: ``reference`` (the verbatim c5ce538 expression -- for cross-checking).
BACKEND = os.environ.get("PSCALIB_CALIB_BACKEND", "auto")


# ==========================================================================
# 1. The memo
# ==========================================================================
_MEMO = {}
_STATS = {"miss": 0, "hit": 0, "stale": 0, "evict": 0, "uncacheable": 0,
          "noncacheable_alias": 0}


def memo_stats():
    """Snapshot of the memo counters.  ``miss`` is the one to assert on."""
    return dict(_STATS)


def memo_clear():
    """Drop every memo entry and zero the counters."""
    _MEMO.clear()
    for k in _STATS:
        _STATS[k] = 0


def memo_size():
    """Number of live memo entries."""
    return len(_MEMO)


def memo_nbytes():
    """Total bytes of DERIVED arrays the memo is holding alive."""
    tot = 0
    for _refs, val in _MEMO.values():
        if isinstance(val, np.ndarray):
            tot += val.nbytes
    return tot


def memo(sources, tag, compute):
    """Memoize ``compute()`` under ``(ids of sources, tag)``, eviction-safe.

    ``sources`` is a tuple of objects (``None`` entries allowed); ``tag`` is any
    hashable discriminator (put scalar parameters like ``status_bits`` in there,
    never in ``sources`` -- small ints are interned and their ids are useless).

    If a source cannot be weak-referenced, or if the derived value IS one of the
    sources, the result is returned WITHOUT being cached: caching the former
    would mean keying on a bare id (unsound), and caching the latter would make
    the memo hold a STRONG ref to a source array, pinning hundreds of MB forever
    and defeating the weakref eviction.
    """
    key = tuple(id(s) for s in sources) + (tag,)
    ent = _MEMO.get(key)
    if ent is not None:
        refs, val = ent
        ok = True
        for r, s in zip(refs, sources):
            if r is None:
                if s is not None:
                    ok = False
                    break
            elif r() is not s:            # the id was recycled -> stale entry
                ok = False
                break
        if ok:
            _STATS["hit"] += 1
            return val
        _STATS["stale"] += 1
        _MEMO.pop(key, None)

    _STATS["miss"] += 1
    val = compute()

    for s in sources:
        if val is s:
            _STATS["noncacheable_alias"] += 1
            return val

    def _evict(_dead, _key=key):          # closes over ints only, no strong refs
        _STATS["evict"] += 1
        _MEMO.pop(_key, None)

    refs = []
    for s in sources:
        if s is None:
            refs.append(None)
            continue
        try:
            refs.append(weakref.ref(s, _evict))
        except TypeError:                 # not weak-referenceable -> do not cache
            _STATS["uncacheable"] += 1
            return val
    _MEMO[key] = (tuple(refs), val)
    return val


# ==========================================================================
# 2. Derived constants (bit-identical to the reference's own derivation)
# ==========================================================================
def _as_f32(src):
    """``np.asarray(src, np.float32)``, hoisted when it actually converts."""
    a = np.asarray(src, dtype=np.float32)
    if a is src:
        return a
    return memo((src,), "as_f32", lambda: a)


def derive_poff(pedestals, pixel_offset):
    """``pedestals + pixel_offset`` as float32, bit-identical to the reference.

    Reference (c5ce538)::

        if pixel_offset is None: poff = pedestals.copy()
        else: poff = (pedestals + np.asarray(pixel_offset, np.float32)).astype(np.float32)

    Two bit-neutral changes: ``pedestals.copy()`` -> ``pedestals`` itself (poff
    is only ever READ, and the reference never mutates it either, so the 201 MB
    copy is dead weight), and ``.astype(np.float32)`` ->
    ``.astype(np.float32, copy=False)``.  The astype is KEPT rather than deleted
    so an exotic ``pixel_offset`` still yields the same dtype and the same bits;
    because ``pedestals`` is already float32 and ``pixel_offset`` is forced to
    float32 first, the sum is ALWAYS float32 and the copy is always waste.
    """
    if pixel_offset is None:
        return pedestals
    return (pedestals + np.asarray(pixel_offset, dtype=np.float32)
            ).astype(np.float32, copy=False)


def derive_gfac(pixel_gain):
    """Protected ``1/pixel_gain``, bit-identical to the reference.

    The divide is kept EXACTLY as the reference writes it: ``np.divide(1.0, g,
    out=np.zeros_like(g, np.float32), where=g != 0)``.  That is a
    SINGLE-PRECISION divide -- the python ``1.0`` is a weak scalar and the
    ``out=`` array pins the loop to float32.  Computing ``1/g`` in float64 and
    rounding afterwards gives a DIFFERENT float32 for many ``g``.  Only the
    trailing no-op ``.astype(np.float32)`` (which copied 201 MB to change
    nothing) is dropped.
    """
    return np.divide(1.0, pixel_gain,
                     out=np.zeros_like(pixel_gain, dtype=np.float32),
                     where=pixel_gain != 0)


# Mask dtypes for which ``x_f32 *= mask`` (numpy casting the mask up inside the
# ufunc loop) is bit-identical to the reference's
# ``x_f32 *= mask.astype(np.float32)``.  The cast must be EXACT for every
# representable value: otherwise numpy would pick a WIDER common loop dtype and
# round afterwards, which is a different result (e.g. int32 or float64 masks
# promote the multiply to float64 -> DOUBLE ROUNDING).
_EXACT_TO_F32_KINDS = frozenset([
    np.dtype(np.bool_), np.dtype(np.uint8), np.dtype(np.int8),
    np.dtype(np.uint16), np.dtype(np.int16), np.dtype(np.float16),
    np.dtype(np.float32),
])


def _prep_mask(mask, hoist):
    """The mask in a dtype we can multiply by directly, hoisted if converted."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype in _EXACT_TO_F32_KINDS:
        return m
    if hoist:
        return memo((mask,), "mask_f32", lambda: m.astype(np.float32))
    return m.astype(np.float32)


# ==========================================================================
# 3. The OPTIONAL fused numba backend
# ==========================================================================
# ``import pscalib`` MUST work with numpy alone, so numba is imported here
# behind try/except and every failure mode degrades to the numpy hybrid.
#   * cache=False  -- cache=True would race across the 16 Ray worker processes
#                     all writing __pycache__ into the installed worktree.
#   * fastmath=False -- fastmath licenses reassociation and FMA contraction,
#                     both of which break bit-exactness.
#   * parallel=False -- pre-committed integrity rule: single-threaded only.
NUMBA_AVAILABLE = False
NUMBA_IMPORT_ERROR = None
NUMBA_VERSION = None
try:
    import numba as _numba
    from numba import njit as _njit
    NUMBA_AVAILABLE = True
    NUMBA_VERSION = getattr(_numba, "__version__", "?")
except ImportError as _exc:                                # pragma: no cover
    NUMBA_IMPORT_ERROR = "ImportError: %s" % (_exc,)
except Exception as _exc:                                  # pragma: no cover
    NUMBA_IMPORT_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

#: raw dtypes the fused kernel may handle.  Restricted on purpose: for a uint64
#: raw, ``v & 0x3fff`` promotes to float64 under numba's typing rules, which
#: would silently reintroduce double rounding.  Anything else falls back to the
#: numpy hybrid, which handles every dtype.
_NUMBA_RAW_DTYPES = frozenset(np.dtype(t) for t in
                              (np.uint8, np.int8, np.uint16, np.int16,
                               np.uint32, np.int32, np.int64))
#: mask dtypes the fused kernel may handle (all convert to float32 by a single
#: correctly-rounded cast, exactly as the reference's ``.astype(np.float32)``).
_NUMBA_MASK_DTYPES = frozenset(np.dtype(t) for t in
                               (np.bool_, np.uint8, np.int8, np.uint16,
                                np.int16, np.uint32, np.int32, np.int64,
                                np.float16, np.float32, np.float64))

_fused_seg = None
_fused_seg_masked = None

if NUMBA_AVAILABLE:                                        # pragma: no branch

    @_njit(cache=False, fastmath=False, parallel=False, nogil=True,
           boundscheck=False)
    def _fused_seg(arr, po0, po1, po2, gf0, gf1, gf2, out):     # noqa: F811
        """One segment, NO mask.  EVERY intermediate is explicitly float32.

        ``code = (v >> 14) & 0xFF`` reproduces the reference's
        ``(arr >> BSH).astype(np.uint8)`` truncation EXACTLY -- including for a
        signed raw word (arithmetic shift then wrap to 8 bits) and for a wide
        code such as 0x100, which the reference collapses to stage 0.  Codes
        ``0`` / ``1`` / ``3`` select stages 0 / 1 / 2; ANYTHING else is the BAD
        code and takes the reference's ``np.select`` default lane, i.e. it
        COMPUTES ``(adc - 0.0) * 0.0`` -- never a literal zero -- so the sign of
        the zero is decided by the multiply.

        This is deliberately ONE FLAT LOOP over the segment.  Two more elaborate
        shapes were built and MEASURED, and BOTH were slower, so both were
        discarded (see IMPL_PROGRESS.txt, jobs 34301838 / 34302162):

          * classifying blocks of columns inside the kernel with an OR-scan and
            dispatching to a specialised loop per block: the scalar OR chain
            alone cost as much as the whole stage-0 pass;
          * the same dispatch with the classifier PRECOMPUTED by a vectorised
            ``np.bitwise_or.reduce`` pre-pass: still 1.7x slower than this flat
            loop even at one block per row (46.5 vs 25.3 ms on a pure-stage-0
            frame, normalised against the numpy path on the same node).  The
            nested block loop is what costs it -- numba stops unrolling.

        So the three-case dispatch lives in the NUMPY path, where the reductions
        are vectorised and the specialised passes are whole-array ufuncs; here a
        single tight loop wins.

        ``np.float32(0)`` -- with an INT literal -- not ``np.float32(0.0)``: a
        python float literal is a float64 constant, and although numba
        immediately narrows it, it makes the word appear in the annotated types
        and so defeats the structural assertion below.  An int literal converts
        to the identical ``+0.0f``.
        """
        zero = np.float32(0)
        nrow, ncol = arr.shape
        for r in range(nrow):
            for c in range(ncol):
                v = arr[r, c]
                code = (v >> 14) & 0xFF
                x = np.float32(v & 0x3fff)
                if code == 0:
                    x = x - po0[r, c]
                    x = x * gf0[r, c]
                elif code == 1:
                    x = x - po1[r, c]
                    x = x * gf1[r, c]
                elif code == 3:
                    x = x - po2[r, c]
                    x = x * gf2[r, c]
                else:
                    x = x - zero
                    x = x * zero
                out[r, c] = x

    @_njit(cache=False, fastmath=False, parallel=False, nogil=True,
           boundscheck=False)
    def _fused_seg_masked(arr, po0, po1, po2, gf0, gf1, gf2, msk, out):  # noqa: F811,E501
        """One segment, WITH the mask.  See :func:`_fused_seg` for the decode.

        The mask multiply keeps the reference's operand order (``arrf *= maskf``
        -> the running value is the DESTINATION), which is what decides which NaN
        payload survives on x86, and it is NEVER short-circuited: a masked pixel
        whose value is finite-negative must come out as ``-0.0``, and a masked
        pixel under a non-finite mask must come out as NaN.
        """
        zero = np.float32(0)
        nrow, ncol = arr.shape
        for r in range(nrow):
            for c in range(ncol):
                v = arr[r, c]
                code = (v >> 14) & 0xFF
                x = np.float32(v & 0x3fff)
                if code == 0:
                    x = x - po0[r, c]
                    x = x * gf0[r, c]
                elif code == 1:
                    x = x - po1[r, c]
                    x = x * gf1[r, c]
                elif code == 3:
                    x = x - po2[r, c]
                    x = x * gf2[r, c]
                else:
                    x = x - zero
                    x = x * zero
                out[r, c] = x * np.float32(msk[r, c])


def _numba_usable(raw, mprep, poff, gfac):
    if not NUMBA_AVAILABLE:
        return False
    if np.dtype(raw.dtype) not in _NUMBA_RAW_DTYPES:
        return False
    if poff.dtype != np.float32 or gfac.dtype != np.float32:
        return False
    if mprep is not None and np.dtype(mprep.dtype) not in _NUMBA_MASK_DTYPES:
        return False
    return True


def _annotation_lines(fn):
    """The TYPE-ANNOTATION lines of ``inspect_types`` output, source echo dropped.

    ``inspect_types`` interleaves the function's own source (which here includes
    docstrings that talk ABOUT float64) with numba's annotations, and every
    annotation line begins with ``#``.  Scanning the raw text would match the
    prose, so filter to the annotations first.
    """
    import io
    buf = io.StringIO()
    fn.inspect_types(file=buf)
    return [ln.strip() for ln in buf.getvalue().splitlines()
            if ln.strip().startswith("#")]


def _float64_lines(fn):
    """Annotated-type lines of every compiled specialization mentioning float64.

    Split into (hard, benign).  BENIGN is exactly one shape: a ``float64`` that
    is the ARGUMENT of an explicit narrowing cast (``(float64,) -> float32``, the
    ``np.float32(...)`` constructor) or the declared type of a float64 *input
    array* (a float64 ``mask``, which the reference itself narrows with
    ``.astype(np.float32)``) or a ``getitem`` reading one element out of it.
    Everything else is HARD: a float64 that participates in arithmetic, which is
    exactly the double-rounding bug.
    """
    hard, benign = [], []
    for s in _annotation_lines(fn):
        if "float64" not in s:
            continue
        if ("-> float32" in s                      # narrowing cast
                or "array(float64" in s            # a float64 INPUT array
                or "static_getitem" in s or "getitem" in s):
            benign.append(s)
        else:
            hard.append(s)
    return hard, benign


def assert_kernels_are_float32_only(verbose=False):
    """STRUCTURALLY prove no float64 leaked into the fused kernels.

    Compiles both kernels and scans every annotated type numba recorded
    (``inspect_types``) for ``float64``.  This catches the double-rounding bug
    WITHOUT needing the right data, which matters because Sterbenz' lemma makes
    ``(a - b)`` exact -- and therefore hides the bug -- on ~80% of real pixels.

    Two assertions, the first strictly stronger than the second:

      1. on the CANONICAL specialization (uint16 raw, uint8 mask -- what the
         detector and the derived status mask actually are) the annotated types
         must contain the string ``float64`` **exactly zero times**;
      2. on every specialization, no float64 may appear anywhere except as the
         argument of an explicit narrowing cast or as a float64 input array (a
         float64 ``mask``, which the reference narrows too).

    Raises ``AssertionError`` on violation; otherwise returns a report dict.
    """
    if not NUMBA_AVAILABLE:
        return {"numba": False, "reason": NUMBA_IMPORT_ERROR}
    import io
    raw = np.array([[0, 1 << 14, 3 << 14, 2 << 14]], dtype=np.uint16)
    pl = [np.zeros((1, 4), np.float32) for _ in range(3)]
    gl = [np.ones((1, 4), np.float32) for _ in range(3)]
    out = np.empty((1, 4), np.float32)
    msk = np.ones((1, 4), np.uint8)
    # exercise ALL THREE dispatch cases (the tiny raw carries codes 0/1/2/3 so
    # block 0 lands in the general case; blk=1 forces per-pixel classification,
    # which reaches the hi==0 and hi==0x4000 branches too).
    _fused_seg(raw, pl[0], pl[1], pl[2], gl[0], gl[1], gl[2], out)
    _fused_seg_masked(raw, pl[0], pl[1], pl[2], gl[0], gl[1], gl[2], msk, out)

    report = {"numba": True, "numba_version": NUMBA_VERSION}
    for name, fn in (("_fused_seg", _fused_seg),
                     ("_fused_seg_masked", _fused_seg_masked)):
        ann = _annotation_lines(fn)
        n_all = sum(ln.count("float64") for ln in ann)
        hard, benign = _float64_lines(fn)
        sigs = [str(s) for s in fn.nopython_signatures]
        report[name] = {
            "n_signatures": len(sigs), "signatures": sigs,
            "n_annotated_lines": len(ann),
            "n_float64_occurrences_total": n_all,
            "n_hard_float64_lines": len(hard),
            "n_benign_float64_lines": len(benign),
            "hard_float64_lines": hard[:20],
            "benign_float64_lines": benign[:6] if verbose else [],
        }
        assert not hard, (
            "float64 LEAKED into numba kernel %s -- that is DOUBLE ROUNDING; "
            "the reference rounds THREE times in float32.  Offending annotated "
            "types:\n%s" % (name, "\n".join(hard[:20])))
        assert all("float64" not in s for s in sigs), (
            "float64 in a %s signature: %s" % (name, sigs))
        # assertion (1): the canonical specialization must be float64-FREE.
        if len(sigs) == 1:
            assert n_all == 0, (
                "the canonical (uint16 raw / uint8 mask) specialization of %s "
                "mentions float64 %d times:\n%s"
                % (name, n_all, "\n".join((hard + benign)[:20])))
        report[name]["canonical_float64_free"] = (len(sigs) == 1 and n_all == 0)
    return report


def backend_info():
    """What the fast path will actually use, and why."""
    return {
        "backend": BACKEND,
        "tile_rows": TILE_ROWS,
        "dense_frac": DENSE_FRAC,
        "numba_available": NUMBA_AVAILABLE,
        "numba_version": NUMBA_VERSION,
        "numba_import_error": NUMBA_IMPORT_ERROR,
        "numpy_version": np.__version__,
    }


# ==========================================================================
# 4. The pure-numpy hybrid kernel
# ==========================================================================
def _gather_fixup(a, resid, poff, gfac, mt, t, s, r0, r1):
    """Overwrite the pixels selected by ``resid`` with their exact per-stage value.

    ``resid`` is a SUPERSET-safe selection: EVERY selected pixel -- including one
    whose truncated gain code turns out to be 0 -- is recomputed from scratch
    here and overwritten WHOLESALE, so whatever the base pass left there is
    discarded and the caller does not have to reason about which base pass ran.
    (Handling code 0 here rather than "leaving it alone" is load-bearing: the
    dense two-plane blend gives the STAGE-1 value to every pixel with a high bit
    set, and a word with e.g. bit 22 set has a high bit yet a code that truncates
    to 0, so "leave code 0 alone" would silently keep the stage-1 value.)
    """
    idx = np.flatnonzero(np.asarray(resid).ravel())
    if idx.size == 0:
        return 0
    af = np.asarray(a).ravel()
    # NOTE the .astype(np.uint8): the reference truncates the gain code to 8
    # bits, so e.g. a code of 256 collapses to 0 (= stage 0).  We must truncate
    # identically or we would "fix up" a pixel the reference treats as G0.
    codes = (af[idx] >> BSH).astype(np.uint8)
    flat_out = t.flags["C_CONTIGUOUS"]
    tf = t.reshape(-1) if flat_out else None
    rr = cc = None
    if not flat_out:
        rr, cc = np.unravel_index(idx, t.shape)
    mf = None if mt is None else np.asarray(mt).ravel()
    m0 = codes == 0
    m1 = codes == 1
    m3 = codes == 3
    # codes 0 / 1 / 3 are stages 0 / 1 / 2; ANYTHING ELSE (2, and any wider code)
    # takes the reference's np.select DEFAULT lane -- the BAD code.
    mbad = ~(m1 | m3 | m0)
    for sel, st in ((m0, 0), (m1, 1), (m3, 2), (mbad, None)):
        ii = idx[sel]
        if ii.size == 0:
            continue
        v = (af[ii] & MSK).astype(np.float32)
        if st is None:
            # NOT short-circuited to a literal 0.0: the reference's np.select
            # gives the bad code pedoff=0 and factor=0, i.e. it COMPUTES
            # (adc_f32 - 0.0) * 0.0, and that multiply decides the sign of the
            # zero (and yields NaN for a non-finite mask).
            v -= np.float32(0.0)
            v *= np.float32(0.0)
        else:
            v -= poff[st, s, r0:r1].ravel()[ii]
            v *= gfac[st, s, r0:r1].ravel()[ii]
        if mf is not None:
            v *= mf[ii]
        if flat_out:
            tf[ii] = v
        else:
            t[rr[sel], cc[sel]] = v
    return int(idx.size)


def _hybrid_numpy(raw, poff, gfac, mprep, out, step, dense_frac):
    """The pure-numpy hybrid.  Returns the per-case tile census."""
    nseg, nrows, ncols = raw.shape
    if not step or step <= 0 or step > nrows:
        step = nrows
    p0a, p1a = poff[0], poff[1]
    g0a, g1a = gfac[0], gfac[1]
    # Scratch, allocated ONCE and reused by every tile so it stays hot in cache
    # instead of being malloc'd and streamed per tile.
    adcb = np.empty((step, ncols), dtype=np.float32)
    x1b = np.empty((step, ncols), dtype=np.float32)
    thr = dense_frac * step * ncols
    census = {"A_pure_g0": 0, "B_dense_blend": 0, "C_sparse_gather": 0,
              "n_gathered": 0}
    # The throwaway (non-G0) lanes of the stage-0 base pass evaluate things like
    # v - NaN and can overflow float32; numpy's default is to WARN.  Suppressing
    # the warning changes no value.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore",
                     under="ignore"):
        for s in range(nseg):
            arr = raw[s]
            p0s, p1s = p0a[s], p1a[s]
            g0s, g1s = g0a[s], g1a[s]
            ms = None if mprep is None else mprep[s]
            osg = out[s]
            for r0 in range(0, nrows, step):
                r1 = r0 + step
                if r1 > nrows:
                    r1 = nrows
                a = arr[r0:r1]
                t = osg[r0:r1]
                mt = None if ms is None else ms[r0:r1]

                # ONE bitwise-OR reduction classifies the tile.  OR commutes
                # with the shift and can only SET bits, so ``hi`` is an exact
                # statement about EVERY pixel in the tile:
                #   hi == 0      -> no bit above bit 13 is ever set: every gain
                #                   code is 0 AND ``raw & MSK == raw``.
                #   hi == 0x4000 -> the only high bit ever set is bit 14: every
                #                   gain code is 0 or 1.
                # (``a.max()`` would NOT do: a negative word of a signed raw
                # dtype has a max below 0x4000 yet a nonzero truncated gain
                # code, and a word with bit 22 set truncates to code 0 yet is
                # NOT equal to ``raw & MSK``.)
                hi = int(np.bitwise_or.reduce(a, axis=None)) & ~MSK
                if hi == 0:
                    # ---- case A: no gain code anywhere in this tile ---------
                    # For a stage-0 pixel raw IS the adc code, so `& MSK` is a
                    # no-op and is dropped.  One constant plane, one pass.
                    census["A_pure_g0"] += 1
                    t[...] = a
                    t -= p0s[r0:r1]
                    t *= g0s[r0:r1]
                    if mt is not None:
                        t *= mt
                    continue

                sel1 = a >= 0x4000
                n1 = int(np.count_nonzero(sel1))
                h = r1 - r0
                if n1 > thr:
                    # ---- case B: DENSE two-plane blend ---------------------
                    # Compute the stage-0 AND stage-1 results for EVERY pixel and
                    # SELECT between the finished values.  np.copyto selects, it
                    # does not compute, so each pixel still receives the
                    # reference's own three float32 ops in the reference's order.
                    # Nothing here is proportional to how many pixels switched:
                    # this is what caps the fixup slope.
                    census["B_dense_blend"] += 1
                    adc, x1 = adcb[:h], x1b[:h]
                    adc[...] = a & MSK
                    x1[...] = adc
                    x1 -= p1s[r0:r1]
                    x1 *= g1s[r0:r1]
                    t[...] = adc
                    t -= p0s[r0:r1]
                    t *= g0s[r0:r1]
                    np.copyto(t, x1, where=sel1)
                    if mt is not None:
                        t *= mt
                    if hi == 0x4000:
                        continue        # only stages 0/1 present: done exactly
                    # The blend handed the stage-1 value to EVERY pixel with a
                    # high bit set, so the residual is every such pixel that is
                    # not truly gain code 1 -- not merely ``a >= 0x8000``: a word
                    # with bit 22 set has ``a >= 0x4000`` (so the blend gave it
                    # stage 1) but its code truncates to 0 in the reference.
                    resid = sel1 & (np.right_shift(a, BSH) != 1)
                else:
                    # ---- case C: stage-0 pass + SPARSE gather --------------
                    census["C_sparse_gather"] += 1
                    t[...] = a & MSK
                    t -= p0s[r0:r1]
                    t *= g0s[r0:r1]
                    if mt is not None:
                        t *= mt
                    resid = sel1
                census["n_gathered"] += _gather_fixup(
                    a, resid, poff, gfac, mt, t, s, r0, r1)
    return census


def _fused_numba(raw, poff, gfac, mprep, out, blk=None):
    """Drive the fused kernel segment by segment (spatial blocking only).

    ``blk`` is accepted and IGNORED: it selected the per-column-block dispatch
    that measurement rejected (see :func:`_fused_seg`).  Kept in the signature so
    the campaign's sweep scripts still run and report the same number.
    """
    nseg = raw.shape[0]
    p0, p1, p2 = poff[0], poff[1], poff[2]
    g0, g1, g2 = gfac[0], gfac[1], gfac[2]
    if mprep is None:
        for s in range(nseg):
            _fused_seg(raw[s], p0[s], p1[s], p2[s], g0[s], g1[s], g2[s], out[s])
    else:
        for s in range(nseg):
            _fused_seg_masked(raw[s], p0[s], p1[s], p2[s],
                              g0[s], g1[s], g2[s], mprep[s], out[s])
    return {"numba_segments": nseg}


# ==========================================================================
# 5. The entry point
# ==========================================================================
#: Diagnostics from the LAST call (tile census / which backend actually ran).
LAST_CALL = {}


def calib_jungfrau_fast(raw, pedestals, pixel_gain, pixel_offset=None,
                        mask=None, tile_rows=None, dense_frac=None,
                        backend=None, hoist=True, out=None, block_cols=None):
    # ``block_cols`` is accepted and ignored (see _fused_numba); it selected a
    # kernel shape that measurement rejected.
    """Bit-identical fast twin of ``pscalib.apply.jungfrau.calib_jungfrau``.

    The extra keyword arguments are tuning / diagnostic knobs only; the public
    ``calib_jungfrau`` never passes them, so its signature and semantics are
    untouched.
    """
    ped_src, gain_src, off_src = pedestals, pixel_gain, pixel_offset
    if backend is None:
        backend = BACKEND
    if tile_rows is None:
        tile_rows = TILE_ROWS
    if dense_frac is None:
        dense_frac = DENSE_FRAC

    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(
            "raw must be 3-D (N,512,1024); got shape %s" % (raw.shape,))
    pedestals = _as_f32(ped_src) if hoist else np.asarray(ped_src, np.float32)
    pixel_gain = _as_f32(gain_src) if hoist else np.asarray(gain_src, np.float32)
    if pedestals.shape[0] != N_GAIN_STAGES:
        raise ValueError(
            "pedestals leading axis must be %d gain stages; got shape %s"
            % (N_GAIN_STAGES, pedestals.shape))

    # ---- derived constants, HOISTED -------------------------------------
    if hoist:
        poff = memo((ped_src, off_src), "poff",
                    lambda: derive_poff(pedestals, pixel_offset))
        gfac = memo((gain_src,), "gfac", lambda: derive_gfac(pixel_gain))
    else:
        poff = derive_poff(pedestals, pixel_offset)
        gfac = derive_gfac(pixel_gain)
    mprep = _prep_mask(mask, hoist)

    nseg = raw.shape[0]
    # np.empty is byte-identical to np.zeros ONLY because every element is
    # assigned: the loop runs s = 0..nseg-1 with nseg == raw.shape[0], and for
    # each s the tile loop covers rows 0..nrows in contiguous non-overlapping
    # tiles, each assigning its full (rows, ncol) slice.  The sparse fixups only
    # OVERWRITE already-assigned elements.  (Proved by the gate's poison test.)
    if out is None:
        out = np.empty(raw.shape, dtype=np.float32)

    if backend not in ("auto", "numpy", "numba"):
        # 'reference' is a knob of the PUBLIC calib_jungfrau (which routes to
        # calib_jungfrau_reference); reaching here with it would silently run the
        # numpy hybrid and mislabel the result, so refuse instead.
        raise ValueError(
            "backend must be 'auto', 'numpy' or 'numba'; got %r "
            "(use pscalib.apply.jungfrau.calib_jungfrau_reference for the "
            "verbatim reference)" % (backend,))
    used = backend
    if backend == "auto":
        used = "numba" if _numba_usable(raw, mprep, poff, gfac) else "numpy"
    elif backend == "numba" and not _numba_usable(raw, mprep, poff, gfac):
        raise RuntimeError(
            "backend='numba' requested but unusable here (numba_available=%s, "
            "raw.dtype=%s, mask.dtype=%s)"
            % (NUMBA_AVAILABLE, raw.dtype,
               None if mprep is None else mprep.dtype))

    if used == "numba":
        try:
            info = _fused_numba(raw, poff, gfac, mprep, out, block_cols)
        except Exception as exc:                       # pragma: no cover
            if backend == "numba":
                raise
            used = "numpy"
            info = _hybrid_numpy(raw, poff, gfac, mprep, out,
                                 int(tile_rows), float(dense_frac))
            info["numba_fallback_reason"] = "%s: %s" % (type(exc).__name__, exc)
    else:
        info = _hybrid_numpy(raw, poff, gfac, mprep, out,
                             int(tile_rows), float(dense_frac))

    LAST_CALL.clear()
    LAST_CALL.update(info)
    LAST_CALL["backend_used"] = used
    LAST_CALL["tile_rows"] = int(tile_rows)
    LAST_CALL["dense_frac"] = float(dense_frac)
    return out
