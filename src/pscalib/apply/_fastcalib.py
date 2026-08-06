"""pscalib.apply._fastcalib -- the fast numpy-only jungfrau calibration path.

This module is the machinery behind
:func:`pscalib.apply.jungfrau.calib_jungfrau`.  It computes, for every pixel,
LITERALLY the reference expression

    ``((raw & MSK).astype(f32) - poff[stage]) * gfac[stage] * mask.astype(f32)``

with the reference's operation order and its THREE separate float32 roundings,
so its output is byte-identical to the reference's -- including the sign of
every zero and every NaN payload -- while doing far less work per event.

NUMPY IS THE ONLY BACKEND.  There is no JIT, no ahead-of-time compiler and no
compiled extension of any kind here -- not even behind a ``try/except`` -- so
``import pscalib`` needs numpy and the python stdlib and nothing else.
``PSCALIB_CALIB_BACKEND`` accepts ``auto``, ``numpy`` and ``reference``; ANY
other value raises ``ValueError`` naming the value asked for, and never falls
back silently, because a silent fallback would mislabel a measurement.

Two things carry the speedup, and neither changes a single output bit:

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

2. **The tile loop (the pure-numpy hybrid kernel).**  Each segment is walked in
   spatial row tiles of :data:`TILE_ROWS` rows, and per tile one of three
   strategies is chosen from ONE cheap pass over that tile.  The classifier is
   the boolean ``sel = (tile >= 0x4000)`` and its ``count_nonzero``; the SAME
   buffer is then reused as the residual selector and as the dense/sparse
   decision, so nothing is scanned twice:

     A. ``count == 0`` -- no pixel has any bit above bit 13, so every gain code
        is 0 AND ``raw & 0x3fff == raw``: the whole tile is stage 0 and the
        ``& MSK`` can be DROPPED.  One constant plane, one streaming pass.
     B. many non-G0 pixels -- a DENSE two-plane blend: compute the stage-0 AND
        the stage-1 result for EVERY pixel and ``np.copyto(..., where=)`` between
        the finished values.  Selection selects, it does not compute, so this is
        bit-exact; and nothing in it is proportional to how many pixels switched,
        which is what caps the fixup slope on high-non-G0 frames.
     C. few non-G0 pixels -- the stage-0 pass, then a SPARSE gather fixup that
        overwrites just those pixels.  Here the ``& MSK`` is DROPPED TOO: for
        UNSIGNED raw every word carrying a bit above bit 13 satisfies
        ``a >= 0x4000``, is therefore in the residual set, and is overwritten
        WHOLESALE by the gather; every other word satisfies ``a & MSK == a``.

   B-vs-C is decided per tile by the tile's own non-G0 count against
   :data:`DENSE_FRAC` (measured, not guessed).  Either way stage 2 (``gbits==3``)
   and the BAD code (``gbits==2``) -- together well under 0.002% of pixels -- are
   gathered.

   Every one of those classifiers is sound ONLY FOR UNSIGNED raw, which is why
   ``raw.dtype.kind != "u"`` is routed to the verbatim reference one function
   below (:func:`calib_jungfrau_fast`) and never reaches this kernel.  For a
   SIGNED dtype a negative word has a small ``max`` and compares False against
   ``0x4000``, yet its truncated gain code is 255 -- the reference's BAD lane --
   so it would silently keep a stage-0 value.  Given unsigned raw,
   ``count_nonzero(a >= 0x4000) == 0`` is EXACTLY the older
   ``bitwise_or.reduce(a) & ~MSK == 0`` (both say "no bit above bit 13 is set
   anywhere in this tile"), and, in the dense branch, ``a.max() < 0x8000`` is
   EXACTLY the older ``hi == 0x4000`` (both say "no bit above bit 14 is set
   anywhere in this tile", i.e. only stages 0 and 1 are present).

   Both branches evaluate the expression at the top of this docstring for every
   pixel, in the reference's operation order and with its three float32
   roundings -- EXCEPT that, when :func:`fold_is_exact` says it is a theorem,
   the last two multiplies are pre-composed into ``gfm = gfac * mask`` once per
   (gain, mask) pair and the per-tile ``t *= mask`` disappears (see
   :func:`fold_is_exact` for the proof and the fail-closed guard).  The path
   choice depends only on the frame's own contents, so it is invariant under any
   event partition; blocking is over SPATIAL axes only.

Two environment knobs tune the tile loop.  BOTH ARE READ ONCE, AT IMPORT TIME,
into module constants, so setting them after ``import pscalib`` has no effect:

  * ``PSCALIB_CALIB_TILE_ROWS`` (int, default 512) -> :data:`TILE_ROWS`: rows
    per spatial tile.
  * ``PSCALIB_CALIB_DENSE_FRAC`` (float, default 0.60) -> :data:`DENSE_FRAC`:
    the per-tile non-G0 fraction above which strategy B is taken instead of C.
    ``0.0`` forces B on every tile; any value above 1 forces C on every tile.

Both are SPEED-ONLY -- the output is byte-identical at every setting, which the
bit gate checks over tile_rows {32, 128, 512} x dense_frac {0.0, 0.60, 1e9}.
See :data:`TILE_ROWS` and :data:`DENSE_FRAC` for the measurements behind the
two defaults.

Traps that are deliberately respected here (each one has already bitten):

  * ``-0.0`` IS LOAD-BEARING.  A masked pixel whose ``(adc - poff) * gfac`` is
    finite-negative becomes ``x * 0.0 == -0.0``.  Writing a literal ``0.0`` for
    masked pixels is BIT-WRONG.  Nothing here ever short-circuits the mask
    multiply -- and the ``gfm`` fold, which LOOKS like a short-circuit, is
    guarded by :func:`fold_is_exact` precisely so that it reproduces that sign.
  * NO DOUBLE ROUNDING.  The reference rounds THREE times (subtract, multiply,
    mask-multiply).  Computing the chain in float64 and storing once gives ONE
    rounding and differs on millions of pixels per frame.  Every intermediate
    here is float32, and every scalar literal is written ``np.float32(...)``
    because a bare python float is a float64 that would promote the whole
    expression.
  * THE BAD CODE'S SIGN.  ``gbits == 2`` has no gain stage; the reference's
    ``np.select`` default gives it ``pedoff=0``/``factor=0``, i.e. it *computes*
    ``(adc - 0.0) * 0.0``, and that multiply decides the sign of the zero (and
    yields NaN for a non-finite mask).  We compute it the same way.
  * NaN PAYLOADS.  On x86 ``MULSS`` returns the DESTINATION operand when it is
    NaN, so ``m * v`` instead of ``v * m`` can change which payload propagates.
    Operand order is identical to the reference in every multiply.
  * SINGLE-THREADED.  Plain numpy ufuncs on one thread, no thread pool, no
    ``concurrent.futures``, no BLAS call.  A hidden thread would manufacture a
    fake speedup at workers=1.

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
    "derive_poff", "derive_gfac", "derive_gfm", "fold_is_exact",
    "calib_jungfrau_fast", "backend_info",
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


#: Rows per spatial tile for the numpy path.  A tile plus its scratch wants to
#: stay resident in cache while the in-place ops stream over it.
#:
#: 128 was the measured optimum for the PREVIOUS kernel (job 34302162).  The
#: current kernel does strictly fewer passes per tile -- the classifier, the
#: dense/sparse count and the residual selector are ONE pass, the dense scratch
#: is raw-dtype rather than float32, and the mask multiply is folded away -- so
#: the per-tile working set shrank and the per-tile python overhead became the
#: thing worth amortising.  RE-MEASURED end to end over the 8 real fixture
#: frames (job 34316936 on sdfmilan253, dense_frac=0.60, mean ms/frame):
#:   tile_rows   32     64     128    256    512
#:   ms/frame    44.96  39.09  37.72  35.08  34.25   <- 512 wins
#: so the default moves 128 -> 512.  (512 == the full segment height, i.e. one
#: tile per segment, for the 512x1024 jungfrau panel.)  Byte-exactness is
#: tile-INDEPENDENT and the gate proves it at 32, 128 and 512.
TILE_ROWS = _env_int("PSCALIB_CALIB_TILE_ROWS", 512)

#: Per-tile non-G0 FRACTION above which the dense two-plane blend beats the
#: sparse gather.  MEASURED, not guessed: forcing dense-on-every-tile against
#: gather-on-every-tile over a synthetic frac sweep 0 -> 1 (job 34302162, node
#: sdfmilan057, ms/frame at tile_rows=128) crosses over at frac ~ 0.25 --
#:   frac   0.002  0.010  0.030  0.080  0.150  0.250  0.400  0.600  1.000
#:   dense  30.6   35.4   40.8   51.4   58.6   61.9   67.4   74.9   38.1
#:   gather 26.4   33.3   36.8   45.3   56.9   71.0   93.0  119.5  175.3
#: The gather is cheaper below the crossover and its cost grows without bound
#: above it; the dense blend is flat-ish.  But that sweep varied the WHOLE-FRAME
#: fraction while the threshold is applied PER TILE, and gain switching is
#: CLUSTERED, so a frame at 20% carries tiles well above 20% -- so the threshold
#: was then tuned directly, end to end, over {0.02, 0.15, 0.35, 0.60} (job
#: 34302301, mean ms/event over the 8 real frames at tile_rows=128):
#:   dense_frac 0.02   0.15   0.35   0.60
#:   mean ms    60.72  56.88  52.50  50.92     <- 0.60 wins
#: and the same ordering holds at every synthetic fraction, so 0.60 is adopted.
#: Byte-exactness is threshold-INDEPENDENT: the gate forces dense-on-every-tile
#: (0.0) and gather-on-every-tile (1e9) and both are byte-exact on all 8 real
#: frames, so any per-tile mixture of the two is byte-exact too.
#: RE-MEASURED for the current kernel (job 34316936, sdfmilan253, tile_rows=128,
#: mean ms/frame over the 8 real fixture frames):
#:   dense_frac  0.02   0.10   0.20   0.35   0.60   1e9
#:   ms/frame    41.05  44.52  38.50  36.85  35.71  42.09    <- 0.60 still wins
#: so 0.60 is kept.
DENSE_FRAC = _env_float("PSCALIB_CALIB_DENSE_FRAC", 0.60)

#: ``auto`` (== ``numpy``: the numpy hybrid is the ONLY compute backend) |
#: ``numpy`` | ``reference`` (the verbatim c5ce538 expression, handled one level
#: up in :mod:`pscalib.apply.jungfrau` -- for cross-checking).
#: ANY other value -- in particular the name of any JIT or compiled accelerator
#: -- is a hard ``ValueError``; see :func:`calib_jungfrau_fast`.
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
# 3. Backend introspection
# ==========================================================================
#: The compute backends this module implements.  ONE entry, on purpose:
#: ``reference`` is not in here because it is dispatched one level up, in
#: :func:`pscalib.apply.jungfrau.calib_jungfrau`, and never reaches this module.
COMPUTE_BACKENDS = ("numpy",)


def backend_info():
    """What the fast path will actually use, and why.

    ``compute_backend`` is ALWAYS ``"numpy"``: the tiled pure-numpy hybrid of
    section 4 is the only kernel that exists.  ``backend`` echoes the
    ``PSCALIB_CALIB_BACKEND`` request (``auto`` / ``numpy`` / ``reference``).
    """
    return {
        "backend": BACKEND,
        "compute_backend": "numpy",
        "compute_backends": list(COMPUTE_BACKENDS),
        "kernel": "numpy_hybrid",
        "compiled_extensions": [],
        "tile_rows": TILE_ROWS,
        "dense_frac": DENSE_FRAC,
        "numpy_version": np.__version__,
    }


# ==========================================================================
# 4. The pure-numpy hybrid kernel
# ==========================================================================
#: The ``dtype=`` of the fused ``np.subtract(a, p, out=t, dtype=np.float32)``
#: that replaced ``t[...] = a ; t -= p`` in every base pass.  It is
#: BELT-AND-BRACES: a float32 ``out=`` already pins the loop, and the fused form
#: was MEASURED bit-identical to the two-step form on numpy 1.26.4 for uint16,
#: uint32 AND uint64 raw with or without the keyword.  It is written anyway
#: because it makes the single-precision requirement explicit at the call site
#: -- the worry it answers is that for a raw dtype that does not cast SAFELY to
#: float32 numpy might pick the ``dd->d`` loop, subtract in float64 and narrow
#: into ``out``, i.e. DOUBLE-ROUND.  It does not, and this keyword stops a
#: future numpy from re-opening the question silently.  It costs nothing.
_SUB_DTYPE = np.float32

#: Magnitude ceiling for :func:`fold_is_exact`.  See its docstring for why
#: 1e15 leaves ~8 orders of magnitude of headroom under float32's 3.403e38.
FOLD_LIM = 1e15


def fold_is_exact(poff, gfac, mprep, raw_dtype, lim=FOLD_LIM):
    """May ``gfm = gfac * mask`` replace the two multiplies ``(x*gfac)*mask``?

    Returns ``(ok, reason)``.  FAIL-CLOSED: every conjunct must hold, and any
    one that does not returns ``False`` with the reason, never an exception --
    the caller then runs the UNFOLDED path (which keeps the per-tile
    ``t *= mask``) and is byte-exact by the older argument.

    ``(x*g)*m == x*(g*m)`` is NOT an identity; float multiplication is not
    associative.  It IS exact under these conditions:

      1. ``raw`` is UNSIGNED.  The whole kernel's classifiers need this anyway
         (see the module docstring), and the ADC bound in (4) uses it.
      2. ``m`` is EXACTLY 0 or 1, and its zeros are ``+0.0`` (never ``-0.0``).
         Then ``g*m`` is either ``g`` (exact) or ``+-0.0`` carrying ``g``'s
         sign, and ``x*(+-0.0)`` has the sign ``sign(x) XOR sign(g)`` -- which
         is exactly ``sign(x*g)``, so the reference's ``-0.0`` for a
         finite-negative masked pixel is reproduced.
         ``-0.0`` is EXCLUDED even though it satisfies the stage-0/1/2 lanes,
         because it BREAKS THE BAD LANE: the reference gives ``gbits == 2`` the
         ``np.select`` default and computes ``(adc - 0.0) * 0.0``, which for
         ``adc >= 0`` is ``+0.0``, and then multiplies by the mask -- giving
         ``-0.0`` for a ``-0.0`` mask.  The folded gather does NOT multiply the
         bad lane by the mask (it has no gain plane to fold into), so it would
         leave ``+0.0``.  With the mask's zeros pinned to ``+0.0`` the two
         agree, and dropping the bad lane's mask multiply becomes a theorem:
         ``+0.0 * 1 == +0.0 * 0 == +0.0``.
      3. ``x`` and ``g`` are finite (so ``poff`` and ``gfac`` are finite, since
         ``x = adc - poff``), and ``float32 x mask.dtype`` stays float32, so the
         fold introduces no wider intermediate and no extra rounding.
      4. ``x*g`` does not overflow to ``+-inf``.  ``inf*0`` is NaN while
         ``x*(+-0.0)`` is ``+-0.0``, so an overflow would BREAK the fold.  With
         ``|poff| < lim``, ``|gfac| < lim`` and unsigned raw, every pixel whose
         value SURVIVES has ``adc <= 0x3fff == 16383`` -- the base pass's
         dropped ``& MSK`` can leave a larger intermediate, but only for pixels
         that are in the residual set and are then overwritten wholesale by the
         gather, which does apply ``& MSK``.  So ``|x*g| <= (16383 + lim)*lim``;
         at ``lim = 1e15`` that is ~1e30, comfortably below 3.403e38.

    Measured on the real fixture the margins are enormous (max|poff| ~ 1e4,
    max|gfac| ~ 1e1), but the guard is what makes the fold a theorem rather
    than a hope.
    """
    if np.dtype(raw_dtype).kind != "u":
        return False, "raw dtype %s is not unsigned" % np.dtype(raw_dtype)
    if not (np.isfinite(poff).all() and np.isfinite(gfac).all()):
        return False, "poff or gfac is not everywhere finite"
    mp = float(np.abs(poff).max())
    mg = float(np.abs(gfac).max())
    if not (mp < lim and mg < lim):
        return False, ("max|poff|=%.3e max|gfac|=%.3e is not below lim=%.3e"
                       % (mp, mg, lim))
    worst = (np.float32(16383.0) + np.float32(lim)) * np.float32(lim)
    if not np.isfinite(worst):
        return False, "worst-case product (16383+lim)*lim overflows float32"
    if mprep is not None:
        m = np.asarray(mprep)
        if m.shape != tuple(gfac.shape[1:]):
            return False, ("mask shape %s != per-stage gain plane shape %s"
                           % (m.shape, tuple(gfac.shape[1:])))
        if np.result_type(np.float32, m.dtype) != np.dtype(np.float32):
            return False, ("float32 x mask(%s) promotes to %s, not float32"
                           % (m.dtype, np.result_type(np.float32, m.dtype)))
        if m.dtype.kind == "f":
            if not np.isfinite(m).all():
                return False, "mask is not everywhere finite"
            if np.signbit(m).any():
                return False, ("mask carries a negative zero; -0.0 breaks the "
                               "BAD-code lane of the fold (see this "
                               "function's docstring, conjunct 2)")
        if not (m.min() >= 0 and m.max() <= 1):
            return False, ("mask is not within [0,1] (min=%r max=%r)"
                           % (m.min(), m.max()))
        if not np.array_equal(m, m.astype(np.bool_)):
            return False, "mask has values other than exactly 0 or 1"
    return True, ("ok: max|poff|=%.4g max|gfac|=%.4g mask in {+0.0, 1}; worst "
                  "|x*g| <= %.3e < %.3e"
                  % (mp, mg, float(worst), float(np.finfo(np.float32).max)))


def derive_gfm(gfac, mprep):
    """``gfac * mask`` broadcast over the 3 gain stages, as float32.

    ONLY call this when :func:`fold_is_exact` said yes -- it is that predicate
    that makes ``x * gfm`` equal the reference's ``(x * gfac) * mask`` bit for
    bit.  The guard's ``np.result_type`` conjunct is what makes the multiply
    below a float32 loop (a uint8 / bool / float16 mask converts to float32
    EXACTLY for every representable value, so this equals the reference's
    ``gfac * mask.astype(np.float32)``); the ``.astype(..., copy=False)`` is a
    no-op assertion of that, not a cast.
    """
    return (gfac * np.asarray(mprep)[None, ...]).astype(np.float32, copy=False)


def _gather_fixup(a, resid, poff, gfx, mt, t, s, r0, r1, folded=False):
    """Overwrite the pixels selected by ``resid`` with their exact per-stage value.

    ``resid`` is a SUPERSET-safe selection: EVERY selected pixel -- including one
    whose truncated gain code turns out to be 0 -- is recomputed from scratch
    here and overwritten WHOLESALE, so whatever the base pass left there is
    discarded and the caller does not have to reason about which base pass ran.
    (Handling code 0 here rather than "leaving it alone" is load-bearing: the
    dense two-plane blend gives the STAGE-1 value to every pixel with a high bit
    set, and a word with e.g. bit 22 set has a high bit yet a code that truncates
    to 0, so "leave code 0 alone" would silently keep the stage-1 value.)

    ``folded`` says ``gfx`` is ``gfm = gfac * mask`` and already carries the
    mask, so the separate mask multiply is dropped -- for stages 0/1/2 because
    :func:`fold_is_exact` proved it, and for the BAD lane because that lane
    computes ``(adc - 0.0) * 0.0`` with ``adc >= 0``, i.e. ``+0.0``, and
    ``+0.0 * 1 == +0.0 * 0 == +0.0`` (the guard forbids a ``-0.0`` mask, which
    is the only value that would make the dropped multiply observable).
    """
    idx = np.flatnonzero(np.asarray(resid).ravel())
    if idx.size == 0:
        return 0
    af = np.asarray(a).ravel()
    # NOTE the .astype(np.uint8): the reference truncates the gain code to 8
    # bits, so e.g. a code of 256 collapses to 0 (= stage 0).  We must truncate
    # identically or we would "fix up" a pixel the reference treats as G0.
    codes = (af[idx] >> BSH).astype(np.uint8)
    # ``t.reshape(-1)`` on a NON-contiguous view returns a silent COPY, not a
    # view, so every scatter below would land in a temporary and be discarded --
    # with no exception and no warning, and a gate whose own output array is
    # always contiguous could never catch it.  ``out=`` is a PUBLIC parameter
    # now, so a caller really can hand in a strided view; this guard, and the
    # ``np.unravel_index`` fallback it selects, are what make that correct.
    flat_out = t.flags["C_CONTIGUOUS"]
    tf = t.reshape(-1) if flat_out else None
    rr = cc = None
    if not flat_out:
        rr, cc = np.unravel_index(idx, t.shape)
    mf = None if (mt is None or folded) else np.asarray(mt).ravel()
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
            v *= gfx[st, s, r0:r1].ravel()[ii]
        if mf is not None:
            v *= mf[ii]
        if flat_out:
            tf[ii] = v
        else:
            t[rr[sel], cc[sel]] = v
    return int(idx.size)


def _hybrid_numpy(raw, poff, gfx, mprep, out, step, dense_frac, folded=False):
    """The pure-numpy hybrid.  Returns the per-case tile census.

    ``raw`` MUST be unsigned -- every classifier below is only sound for an
    unsigned dtype and :func:`calib_jungfrau_fast` routes anything else to the
    verbatim reference before reaching here.

    ``gfx`` is ``gfac`` when ``folded`` is False and ``gfm = gfac * mask`` when
    it is True; ``folded`` is set ONLY when :func:`fold_is_exact` said so, and
    it is exactly the flag that removes the per-tile ``t *= mask``.
    """
    nseg, nrows, ncols = raw.shape
    if not step or step <= 0 or step > nrows:
        step = nrows
    p0a, p1a = poff[0], poff[1]
    g0a, g1a = gfx[0], gfx[1]
    # Scratch, allocated ONCE and reused by every tile so it stays hot in cache
    # instead of being malloc'd and streamed per tile.  ``adcb`` is in the RAW
    # dtype, not float32: it holds ``a & MSK``, it is half the bytes for uint16
    # raw, and it feeds BOTH dense lanes through a fused subtract -- so the
    # float32 adc copy the previous kernel materialised disappears.  ``selb`` is
    # the classifier's output buffer AND the residual selector AND the
    # ``np.copyto`` mask: one pass over the tile, three uses.
    adcb = np.empty((step, ncols), dtype=raw.dtype)
    x1b = np.empty((step, ncols), dtype=np.float32)
    selb = np.empty((step, ncols), dtype=np.bool_)
    thr = dense_frac * step * ncols
    census = {"A_pure_g0": 0, "B_dense_blend": 0, "C_sparse_gather": 0,
              "n_gathered": 0, "mask_folded": bool(folded)}
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
                h = r1 - r0
                a = arr[r0:r1]
                t = osg[r0:r1]
                mt = None if ms is None else ms[r0:r1]
                p0t, g0t = p0s[r0:r1], g0s[r0:r1]

                # ONE pass over the tile does all THREE jobs: it classifies the
                # tile, it counts for the dense/sparse decision, and it IS the
                # residual selector the gather needs.  For UNSIGNED raw
                # ``count_nonzero(a >= 0x4000) == 0`` is exactly the older
                # ``bitwise_or.reduce(a) & ~MSK == 0``: both say "no bit above
                # bit 13 is set anywhere in this tile", i.e. every gain code is
                # 0 AND ``raw & MSK == raw``.  So the separate reduction over
                # the raw tile disappears.
                sel1 = selb[:h]
                np.greater_equal(a, 0x4000, out=sel1)
                n1 = int(np.count_nonzero(sel1))
                if n1 == 0:
                    # ---- case A: no gain code anywhere in this tile ---------
                    # For a stage-0 pixel raw IS the adc code, so `& MSK` is a
                    # no-op and is dropped.  One constant plane, one pass.
                    census["A_pure_g0"] += 1
                    np.subtract(a, p0t, out=t, dtype=_SUB_DTYPE)
                    t *= g0t
                    if mt is not None and not folded:
                        t *= mt
                    continue

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
                    np.bitwise_and(a, MSK, out=adc)
                    np.subtract(adc, p1s[r0:r1], out=x1, dtype=_SUB_DTYPE)
                    x1 *= g1s[r0:r1]
                    np.subtract(adc, p0t, out=t, dtype=_SUB_DTYPE)
                    t *= g0t
                    np.copyto(t, x1, where=sel1)
                    if mt is not None and not folded:
                        t *= mt
                    # For UNSIGNED raw ``a.max() < 0x8000`` is exactly the older
                    # ``hi == 0x4000``: no bit above bit 14 is set anywhere, so
                    # every gain code is 0 or 1 and the blend is already the
                    # whole answer.  The reduction is evaluated ONLY here, i.e.
                    # on the few dense tiles, never on the common ones.
                    if int(a.max()) < 0x8000:
                        continue        # only stages 0/1 present: done exactly
                    # The blend handed the stage-1 value to EVERY pixel with a
                    # high bit set, so the residual is every such pixel that is
                    # not truly gain code 1 -- not merely ``a >= 0x8000``: a word
                    # with bit 22 set has ``a >= 0x4000`` (so the blend gave it
                    # stage 1) but its code truncates to 0 in the reference.
                    resid = sel1 & (np.right_shift(a, BSH) != 1)
                else:
                    # ---- case C: stage-0 pass + SPARSE gather --------------
                    # The ``& MSK`` is DROPPED here too.  For UNSIGNED raw every
                    # word with a bit above bit 13 satisfies ``a >= 0x4000``, is
                    # therefore in ``resid``, and is overwritten WHOLESALE by
                    # the gather (which does apply ``& MSK``), so whatever this
                    # base pass computed for it -- however large, even an
                    # overflow -- is discarded; and every word that is NOT in
                    # ``resid`` satisfies ``a & MSK == a``.
                    census["C_sparse_gather"] += 1
                    np.subtract(a, p0t, out=t, dtype=_SUB_DTYPE)
                    t *= g0t
                    if mt is not None and not folded:
                        t *= mt
                    resid = sel1
                census["n_gathered"] += _gather_fixup(
                    a, resid, poff, gfx, mt, t, s, r0, r1, folded)
    return census


# ==========================================================================
# 5. The entry point
# ==========================================================================
#: Diagnostics from the LAST call: the per-case tile census, which backend
#: actually ran, the tile/threshold knobs in force, and -- ``mask_folded`` /
#: ``fold_reason`` -- whether the ``gfm = gfac * mask`` fold was taken and, when
#: it was not, the exact conjunct of :func:`fold_is_exact` that declined it.
LAST_CALL = {}


def check_out_buffer(out, shape, dtype=np.float32, name="out"):
    """Validate a caller-supplied ``out=`` buffer; return it, or RAISE.

    The whole point of ``out=`` is that the caller reuses ONE 67 MB buffer
    across events instead of first-touching fresh pages every event.  A
    silently-ignored ``out=`` would therefore be the worst possible failure
    mode: the caller keeps paying the allocation it thought it had removed,
    the result is written somewhere else, and the buffer it later reads holds
    stale data from the previous event -- a WRONG cube, with no error anywhere.
    So every mismatch is a hard error naming what was expected and what came.

    Requirements, all checked:

    * a real ``numpy.ndarray`` (not a list, not a memoryview) -- the kernel
      writes into it with basic slicing and ``ufunc(..., out=)``;
    * exactly ``dtype`` (default float32).  A float64 buffer would make the
      kernel's in-place ``-=`` / ``*=`` round in float64 and CHANGE OUTPUT BITS
      (the reference rounds three times in float32), so an "obviously
      compatible" wider dtype is refused, not upcast into;
    * exactly ``shape``.  Broadcasting is NOT accepted: it would write a
      different array than the one the caller holds;
    * writeable.

    Contiguity is deliberately NOT required: the hybrid kernel writes
    ``out[s][r0:r1]`` slices and its sparse fixup already handles a
    non-contiguous tile (see :func:`_gather_fixup`), so a strided view is
    correct -- merely slower.
    """
    want = np.dtype(dtype)
    shape = tuple(int(x) for x in shape)
    if not isinstance(out, np.ndarray):
        raise TypeError(
            "%s= must be a numpy.ndarray of shape %s and dtype %s; got %s. "
            "(%s= is an optional output buffer: pass None -- the default -- to "
            "let pscalib allocate one.)"
            % (name, shape, want, type(out).__name__, name))
    if out.dtype != want:
        raise ValueError(
            "%s= has dtype %s but must be exactly %s: the calibration rounds in "
            "%s and writing through a wider buffer would change output bits. "
            "Allocate it as np.empty(%s, dtype=np.%s)."
            % (name, out.dtype, want, want, shape, want))
    if out.shape != shape:
        raise ValueError(
            "%s= has shape %s but must be exactly %s (the shape of raw); "
            "broadcasting into a differently-shaped buffer is refused because "
            "the caller would then be reading a different array than the one "
            "that was written." % (name, out.shape, shape))
    if not out.flags.writeable:
        raise ValueError(
            "%s= is not writeable (out.flags.writeable is False); the "
            "calibration writes its result into it." % (name,))
    return out


def calib_jungfrau_fast(raw, pedestals, pixel_gain, pixel_offset=None,
                        mask=None, tile_rows=None, dense_frac=None,
                        backend=None, hoist=True, out=None):
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
    else:
        # A caller-supplied buffer is VALIDATED, never silently ignored and
        # never silently replaced by a fresh allocation -- see
        # :func:`check_out_buffer` for why a quiet fallback would be the worst
        # possible failure mode here.
        out = check_out_buffer(out, raw.shape)

    if backend not in COMPUTE_BACKENDS and backend != "auto":
        # LOUD, never a silent fallback.  Two kinds of caller land here and both
        # must be refused rather than quietly given the numpy hybrid:
        #   * someone asking for a JIT / compiled accelerator by name.  There is
        #     none to fall back FROM (see the module docstring), and a caller who
        #     asked for one and silently got numpy would mislabel every timing
        #     they then published.  The value they asked for is echoed with %r,
        #     so the message names it without this file ever mentioning it.
        #   * 'reference', which is a knob of the PUBLIC calib_jungfrau (it
        #     routes to calib_jungfrau_reference one level up and never reaches
        #     here); running the hybrid for it would mislabel the result too.
        raise ValueError(
            "backend=%r is not a pscalib compute backend.  Valid values are "
            "'numpy' (the tiled pure-numpy hybrid, which is the only compute "
            "backend pscalib has), 'auto' (an alias for 'numpy'), and "
            "'reference' (the verbatim reference expression; select it with "
            "PSCALIB_CALIB_BACKEND=reference, or call "
            "pscalib.apply.jungfrau.calib_jungfrau_reference directly).  "
            "pscalib contains no JIT and no compiled extension, so there is no "
            "accelerator to select; this raises instead of falling back to "
            "numpy so that a timing measurement cannot be mislabelled."
            % (backend,))

    # ---- SIGNED / non-unsigned raw: route to the verbatim reference ------
    # The reference classifies the gain code as ``(arr >> 14).astype(np.uint8)``,
    # so a NEGATIVE word yields 255, which matches none of (0, 1, 3) and
    # therefore falls to the ``np.select`` DEFAULT lane (pedoff 0.0, factor 0.0)
    # and renders as 0.0.  The numpy hybrid's classifier ``a >= 0x4000`` is
    # False for every negative word, so such a pixel never enters the residual
    # set nor the gather fixup and silently keeps its stage-0 base value -- an
    # error of order 3e4 ADU, not a sign-of-zero.
    # Real jungfrau raw is uint16 (the dataset, the fixture and the public
    # docstring all say so), so this was latent and no measured number is
    # affected -- but the pure-numpy path is the byte-exact fallback and must
    # stay one.  Deferring to the reference closes the exactness gap by
    # construction.
    if raw.dtype.kind != "u":
        from .jungfrau import calib_jungfrau_reference
        res = calib_jungfrau_reference(raw, ped_src, gain_src,
                                       pixel_offset=off_src, mask=mask)
        if out is not None:
            out[...] = res
            res = out
        LAST_CALL.clear()
        LAST_CALL.update({
            "backend_used": "reference",
            "reference_reason": "raw.dtype=%s is not unsigned; the fast "
                                "classifiers are only sound for unsigned raw"
                                % (raw.dtype,),
        })
        return res

    # ---- the MASK FOLD, memoised and fail-closed --------------------------
    # ``gfm = gfac * mask`` pre-composes the reference's last TWO multiplies
    # into one plane, which removes a whole streaming multiply from every tile
    # of every event.  It is only taken when :func:`fold_is_exact` proves it is
    # bit-neutral, and a No there is NOT an error -- it selects the unfolded
    # lane, which still carries the per-tile ``t *= mask``.
    #
    # BOTH the predicate and the derived plane are keyed on the IDENTITY of
    # EVERY source they are a function of.  ``gfm`` is a function of
    # (pixel_gain, mask), so it is keyed on BOTH: keyed on the gain alone, a
    # caller who changed ONLY the mask would silently receive the previous
    # mask's fold and the mask would simply be wrong, with no error anywhere.
    # The predicate additionally reads poff (finiteness / magnitude) and the raw
    # dtype, so its key carries pedestals, pixel_offset and the dtype too.
    # (The memo is the eviction-safe weakref one of section 1; an in-place
    # mutation of a source, which does not change its id, is outside its
    # contract -- as it already is for poff, gfac and the status mask.)
    gfx, folded = gfac, False
    if mprep is None:
        fold_reason = "no mask: there is nothing to fold"
    elif not hoist:
        # hoist=False is the un-memoised diagnostic path: folding there would
        # pay a full-size multiply plus two full-array scans on EVERY call.
        fold_reason = "hoist=False: the fold is not worth deriving un-memoised"
    else:
        fold_ok, fold_reason = memo(
            (ped_src, off_src, gain_src, mask), ("fold_is_exact",
                                                 raw.dtype.str, FOLD_LIM),
            lambda: fold_is_exact(poff, gfac, mprep, raw.dtype))
        if fold_ok:
            gfx = memo((gain_src, mask), "gfm",
                       lambda: derive_gfm(gfac, mprep))
            folded = True

    # 'auto' and 'numpy' are the same thing; the numpy hybrid is the only kernel.
    used = "numpy"
    info = _hybrid_numpy(raw, poff, gfx, mprep, out,
                         int(tile_rows), float(dense_frac), folded)

    LAST_CALL.clear()
    LAST_CALL.update(info)
    LAST_CALL["backend_used"] = used
    LAST_CALL["tile_rows"] = int(tile_rows)
    LAST_CALL["dense_frac"] = float(dense_frac)
    LAST_CALL["mask_folded"] = bool(folded)
    LAST_CALL["fold_reason"] = fold_reason
    return out
