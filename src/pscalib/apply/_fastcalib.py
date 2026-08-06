"""pscalib.apply._fastcalib -- the byte-exact fast machinery behind the apply leaves.

NUMPY IS THE ONLY BACKEND.  There is no JIT, no ahead-of-time compiler and no
compiled extension of any kind here -- not even behind a ``try/except`` -- so
``import pscalib`` needs numpy and the python stdlib and nothing else.  An
earlier revision of this file carried an OPTIONAL fused JIT kernel; it was
DELETED for the numpy-only jungfrau cube campaign (branch
``cand/calib-numpy-8x``), which forbids every compiled accelerator anywhere
under ``src/``.  ``PSCALIB_CALIB_BACKEND`` therefore accepts ``auto``, ``numpy``
and ``reference``; ANY other value raises ``ValueError`` naming the value asked
for, and never falls back silently, because a silent fallback would mislabel a
measurement.

This module holds TWO things, neither of which changes a single output bit:

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

Traps that are deliberately respected here (each one has already bitten):

  * ``-0.0`` IS LOAD-BEARING.  A masked pixel whose ``(adc - poff) * gfac`` is
    finite-negative becomes ``x * 0.0 == -0.0``.  Writing a literal ``0.0`` for
    masked pixels is BIT-WRONG.  Nothing here ever short-circuits the mask
    multiply.
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
    "derive_poff", "derive_gfac",
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


# ==========================================================================
# 5. The entry point
# ==========================================================================
#: Diagnostics from the LAST call (tile census / which backend actually ran).
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
                        backend=None, hoist=True, out=None, block_cols=None):
    # ``block_cols`` is accepted and IGNORED.  It used to select a per-column
    # block dispatch inside the deleted fused kernel; the argument is kept so the
    # campaign's existing sweep scripts still run (and now simply report the
    # numpy hybrid's number for every value).
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
            "backend=%r is not available.  The pure-numpy hybrid is the ONLY "
            "compute backend pscalib has: there is no JIT and no compiled "
            "extension anywhere under src/ (branch cand/calib-numpy-8x, the "
            "numpy-only jungfrau cube campaign, forbids every compiled "
            "accelerator).  Use 'numpy', or 'auto', which means the same thing. "
            "For the verbatim c5ce538 expression set "
            "PSCALIB_CALIB_BACKEND=reference or call "
            "pscalib.apply.jungfrau.calib_jungfrau_reference directly.  This "
            "raises instead of falling back so that a measurement cannot be "
            "mislabelled." % (backend,))

    # ---- SIGNED / non-unsigned raw: route to the verbatim reference ------
    # The reference classifies the gain code as ``(arr >> 14).astype(np.uint8)``,
    # so a NEGATIVE word yields 255, which matches none of (0, 1, 3) and
    # therefore falls to the ``np.select`` DEFAULT lane (pedoff 0.0, factor 0.0)
    # and renders as 0.0.  The numpy hybrid's classifier ``a >= 0x4000`` is
    # False for every negative word, so such a pixel never enters the residual
    # set nor the gather fixup and silently keeps its stage-0 base value -- an
    # error of order 3e4 ADU, not a sign-of-zero.
    # Real jungfrau raw is uint16 (the dataset, the fixture and the public
    # docstring all say so), so this was latent and no measured number in this
    # campaign is affected -- but the pure-numpy path is the byte-exact
    # fallback and must stay one.  Deferring to the reference closes the
    # exactness gap by construction.
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

    # 'auto' and 'numpy' are the same thing; the numpy hybrid is the only kernel.
    used = "numpy"
    info = _hybrid_numpy(raw, poff, gfac, mprep, out,
                         int(tile_rows), float(dense_frac))

    LAST_CALL.clear()
    LAST_CALL.update(info)
    LAST_CALL["backend_used"] = used
    LAST_CALL["tile_rows"] = int(tile_rows)
    LAST_CALL["dense_frac"] = float(dense_frac)
    return out
