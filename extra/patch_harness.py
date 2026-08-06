#!/usr/bin/env python
"""Produce bench_cube_calib_fast.py from the UNMODIFIED harness, by exact
string replacement, asserting the source md5 first.

The shared harness file is NEVER edited in place.  This script reads it, checks
its md5 against the pinned value, applies three surgical replacements, and
writes a COPY.  It prints both md5s and a unified diff so the change is
auditable.

The patch is required to be NUMERICALLY INERT: the binned cube the patched
harness produces must be BIT-IDENTICAL to the unmodified harness's, which is
proved separately by cube_inertness.py (both accumulate_slice implementations
run in ONE process over the SAME events and their float64 sums are compared as
raw bit patterns, not with a tolerance).

What the patch changes, and why each change cannot move a bit:

  H1  PREFETCH.  ``ridx.read_events(sub)`` returns a LIST (psdata/index.py:1074
      -- "Returns ... list[psdata.stream.Event] ... in ks order"), so a stdlib
      thread can materialise batch i+1 while the main thread computes batch i.
      A depth-1 FIFO queue preserves batch order exactly, and only the reader
      thread ever touches ``ridx``.  The 38.7 ms/event read then OVERLAPS
      compute instead of adding to it.  No value changes: the same events are
      consumed in the same order.

  H2  REUSED CALIB OUTPUT BUFFER.  ``pscalib.calib(..., out=buf)`` writes into a
      buffer allocated once instead of first-touching 67 MB of fresh pages every
      event.  Support is DETECTED, not assumed: if this pscalib has no ``out=``
      the patch falls back to the plain call and says so on its banner line.

  H3  ALL-FINITE FAST PATH -- data-driven, not assumed.  ``np.isfinite`` writes
      into a reused bool buffer and the event's finite count is taken with
      ``np.count_nonzero``.  ONLY when the count equals ``cal.size`` does the
      fast path run, and then ``np.where(finite, cal, 0.0)`` IS ``cal`` element
      for element (verified: numpy 1.26.4 keeps float32 because the python
      ``0.0`` is a weak scalar, and every lane takes the ``cal`` branch), so
      ``sums[b] += cal`` is bit-identical to ``sums[b] += contrib`` while
      skipping a 67 MB allocation and a 67 MB copy.  Any event with even one
      non-finite pixel falls through to the VERBATIM original expression.
      NOTE the rejected alternative: ``np.add(sums, cal, out=sums, where=finite)``
      is NOT equivalent -- a lane holding -0.0 that the original would have
      incremented by +0.0 becomes +0.0, and ``where=`` leaves it -0.0.

  H4  LAZY nvalid.  ``nvalid[b] += finite`` with an all-True ``finite`` is
      ``nvalid[b] += 1``.  Instead of touching 67 MB of uint32 twice per event,
      the count of all-finite events per bin is kept as a python int and the map
      is materialised once at the end: ``nvalid[b] = partial_map + scalar``.
      Integer addition is associative and exact, so the map is bit-identical.

  H6  REUSED STACK BUFFER.  ``psdata.stream.Event.stack`` (stream.py:560) is
      ``segs = self.raw(...); out = np.empty(...); for k, s in
      enumerate(sorted(segs)): out[k] = segs[s]``.  The patch performs exactly
      that algorithm into a buffer allocated once, using only the public
      ``evt.raw()``.  psdata is NOT modified.  Same segment order, same dtype,
      same copies -- only the 33.5 MB allocation is reused.

--------------------------------------------------------------------------
LEVEL 2 -- ``--level 2`` / bench_cube_calib_fast2.py.  OFF BY DEFAULT.
--------------------------------------------------------------------------
Level 2 is a SUPERSET of level 1 applied by pure INSERTION: it does not delete
or rewrite one character of the level-1 text (``--level 2`` asserts that the
diff level1 -> level2 contains no '-' lines).  At runtime it is gated on
``BENCH_FAST_BATCH``, whose default is 0 = OFF, and with the knob at 0 the
level-2 file executes exactly the level-1 code path -- the batched accumulator
is a separate function that is never called.

  H7  BATCHED float64 ACCUMULATE (``BENCH_FAST_BATCH=K``, K>0).
      The problem: ``sums[b] += cal`` reads AND writes a 134 MB float64 array
      and reads a 67 MB float32 array -- ~335 MB of memory traffic per event --
      and with nbins=10 the bin's accumulator is cold in every cache by the time
      that bin comes round again.
      The move: hold up to K calibrated events as float32 contributions.  When
      the buffer is full (or the slice ends), GROUP the buffered events by bin,
      PRESERVING ARRIVAL ORDER WITHIN EACH GROUP, and for each bin walk SLABS of
      that bin's ``sums`` array, adding the group's contributions to the slab in
      order:
          for slab in slabs:
              s = sums[b][slab]
              for c in group_in_arrival_order:
                  s += c[slab]
      Traffic per event drops from ``K x 335 MB`` to ``K x 67 MB + (bins
      touched) x 268 MB``, and each slab is small enough to stay in L2/L3 for
      the whole inner loop.

      WHY IT IS BIT-IDENTICAL, not merely close:
        (a) Per ELEMENT the sequence of float64 additions is UNCHANGED.  The
            element at index p of bin b receives exactly the same summands in
            exactly the same order as before.  Chunking over the element index
            (the slab loop) cannot reorder a per-element sequence -- it only
            decides when each element's next addition is issued.
        (b) Grouping by bin cannot reorder anything either, because distinct
            bins are DISJOINT accumulator arrays: no element is touched by two
            groups.  Order BETWEEN groups is therefore unobservable.
        (c) float64 += float32 is elementwise ``np.add`` with an exact float32
            -> float64 widening of the operand; there is no reduction, no FMA
            contraction (no multiply exists) and no vectorisation-order
            sensitivity, so a slab view and a whole array give the same bits.
        (d) A bin's FIRST buffered contribution initialises the accumulator.
            ``sums[b] = c.astype(np.float64)`` becomes ``s = np.empty(shape,
            float64)`` followed by ``s[slab] = c[slab]`` for every slab -- the
            same exact widening cast, element for element, over a partition of
            the array that covers it exactly once.
        (e) ``nvalid``, ``nfin_scalar``, ``counts``, ``n_nan`` and ``unbinned``
            are still updated EAGERLY, per event, with the level-1 expressions.
            Nothing about H3 or H4 changes; only the float64 add is deferred.
            In particular the slow lane still computes ``np.where(finite, cal,
            0.0)`` VERBATIM and still does ``nvalid[b] += finite`` at once, so
            ``finite`` is never buffered.

      THE CALIB OUTPUT BUFFER CANNOT BE A SINGLE REUSED ARRAY ANY MORE.  H2's
      one ``cal_buf`` is replaced by a RING/POOL of K buffers with an explicit
      free list: an event pops a free slot, calibrates into it, and the slot
      stays held for as long as the buffered contribution points into it; the
      flush hands every slot back.  A slot is released early ONLY in the slow
      lane, where ``np.where`` is guaranteed to have allocated a brand-new array
      so nothing points at the slot any more (the test is ``contrib is not
      cal``, i.e. identity, not a heuristic).  If ``calib`` ever returned
      something other than the buffer it was handed, the slot simply stays held
      -- conservative, never unsafe.  Slots are allocated lazily, so the number
      of live float32 frames is bounded by K+1 (all-fast: K held slots; all-slow:
      1 recycled slot + K fresh ``np.where`` arrays).

      MEMORY.  Live float32 frames <= (K+1) x 67 MB (4.4 GB at K=64), on top of
      the prefetch's (FAST_PREFETCH+1) x subbatch x 33.5 MB (1.07 GB at the
      defaults) and the per-bin accumulators (134 MB float64 + 67 MB uint32 per
      populated bin).  The patched worker PRINTS the figure on its first event.

      SLAB SIZE.  ``BENCH_FAST_SLAB`` = number of leading-axis entries
      (jungfrau segments) per slab, default 4 = 8.4 MB of float32 read against
      16.8 MB of float64 read+written.  Slabs are taken on axis 0 so every slab
      is a genuine VIEW (a flat reshape could silently copy a non-contiguous
      array and turn ``+=`` into a no-op).  Correctness does not depend on the
      value: any positive slab size partitions the array exactly once.

      FLUSH.  The buffer is flushed when it reaches K, and again after the read
      loop ends -- including the early ``break`` on the prefetch sentinel --
      before the H4 nvalid merge and before the return.  A flush with an empty
      buffer is a no-op, so the extra call is free.  If the loop dies with an
      exception the frame unwinds without a return value and the partial cube is
      discarded by the caller, so there is nothing to flush.
"""
import argparse
import difflib
import hashlib
import os
import sys

ORIG_MD5 = "a29e67e5c453fb5cd31e9184fe0e3f7c"

# --------------------------------------------------------------------------
# (1) imports -- add the two stdlib modules the prefetch needs
# --------------------------------------------------------------------------
OLD_IMPORTS = """import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time

import numpy as np
"""

NEW_IMPORTS = """import argparse
import hashlib
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np

# ==========================================================================
# FAST-HARNESS PATCH (bench_cube_calib_fast.py) -- see patch_harness.py for the
# bit-exactness argument behind every one of these.  NUMERICALLY INERT: the
# binned cube is bit-identical to the unmodified harness's.
# ==========================================================================
#: Batches held in flight by the prefetch thread.  1 == classic double
#: buffering: the reader fills batch i+1 while the main thread computes batch i.
FAST_PREFETCH = int(os.environ.get("BENCH_FAST_PREFETCH", "1"))
#: Master switch, for A/B-ing the patch against itself during development.
#: The proof job always runs with this at its default of 1.
FAST_ACCUM = int(os.environ.get("BENCH_FAST_ACCUM", "1"))


class _Buf(object):
    \"\"\"One reusable ndarray, reallocated only when the shape/dtype changes.\"\"\"

    __slots__ = ("a",)

    def __init__(self):
        self.a = None

    def get(self, shape, dtype):
        a = self.a
        if a is None or a.shape != shape or a.dtype != dtype:
            a = np.empty(shape, dtype=dtype)
            self.a = a
        return a


def _stack_into(evt, det_name, buf):
    \"\"\"``evt.stack(det_name)`` with the output buffer REUSED.

    Byte-for-byte the algorithm of ``psdata.stream.Event.stack``
    (psdata/src/psdata/stream.py:560): take the per-segment dict from the public
    ``evt.raw()``, order the segment ids with ``sorted``, and copy each segment
    into row ``k``.  psdata is not modified and not bypassed -- only the 33.5 MB
    ``np.empty`` is hoisted out of the per-event loop, which removes a
    first-touch page fault over 33.5 MB every event.
    \"\"\"
    segs = evt.raw(det_name)
    if segs is None:
        return None
    seg_ids = sorted(segs)
    sample = segs[seg_ids[0]]
    out = buf.get((len(seg_ids),) + sample.shape, sample.dtype)
    for k, s in enumerate(seg_ids):
        out[k] = segs[s]
    return out


def _calib_supports_out(pscalib):
    \"\"\"Does this pscalib's public ``calib`` accept ``out=``?  DETECTED, never
    assumed: an older pscalib without it must still run, just without H2.\"\"\"
    try:
        import inspect
        return "out" in inspect.signature(pscalib.calib).parameters
    except Exception:                                          # noqa: BLE001
        return False
"""

# --------------------------------------------------------------------------
# (2) the accumulate body
# --------------------------------------------------------------------------
OLD_BODY = '''    import pscalib
    from psdata.index import RunIndex

    ridx = RunIndex.from_dict(index_state)
    r0 = rchar()                   # bytes read BY THIS PROCESS (the worker)
    try:
        sums, nvalid, counts = {}, {}, {}
        n_nan = 0
        unbinned = 0
        for i in range(0, len(ks), subbatch):
            sub = [int(k) for k in ks[i:i + subbatch]]
            sub_bins = bins[i:i + subbatch]
            # US-009 coalesced batch read: preads grouped per chunk file, in
            # ascending offset order.  Returns events in REQUEST order.
            for evt, b in zip(ridx.read_events(sub), sub_bins):
                b = int(b)
                if b < 0:
                    unbinned += 1
                    continue
                raw = evt.stack(det)
                if raw is None:
                    continue                    # detector absent from this event
                # ---- THE REAL CALIBRATION (this is what the old cube omitted)
                cal = pscalib.calib(dettype, raw, constants, config=seg_cfg)
                finite = np.isfinite(cal)
                n_nan += int(cal.size - finite.sum())
                contrib = np.where(finite, cal, 0.0)
                if b not in sums:
                    sums[b] = contrib.astype(np.float64)
                    nvalid[b] = finite.astype(np.uint32)
                    counts[b] = 1
                else:
                    sums[b] += contrib
                    nvalid[b] += finite
                    counts[b] += 1
            # sub + its dgram snapshots drop out of scope here (bounded memory)
        # rchar is per-PROCESS, so the read bytes MUST be measured here, inside
        # the worker -- the driver's own /proc/self/io cannot see a Ray worker's
        # reads at all (separate processes), so a driver-side rchar delta would
        # report ~0 MB/s for every parallel run.
        return sums, nvalid, counts, n_nan, unbinned, rchar() - r0
    finally:
        ridx.close()
'''

NEW_BODY = '''    import pscalib
    from psdata.index import RunIndex

    ridx = RunIndex.from_dict(index_state)
    r0 = rchar()                   # bytes read BY THIS PROCESS (the worker)
    if not FAST_ACCUM:
        return _accumulate_slice_orig(ridx, r0, ks, bins, det, dettype,
                                      constants, seg_cfg, subbatch, pscalib)
    reader = None
    stop = threading.Event()
    q = queue.Queue(maxsize=max(1, FAST_PREFETCH))
    try:
        sums, nvalid, counts = {}, {}, {}
        # H4: per-bin count of events whose calib output was ENTIRELY finite.
        # For those events ``nvalid[b] += finite`` is ``nvalid[b] += 1``, so the
        # 67 MB uint32 map need not be touched at all until the very end.
        nfin_scalar = {}
        n_nan = 0
        unbinned = 0
        use_out = _calib_supports_out(pscalib)
        cal_buf = _Buf()
        fin_buf = _Buf()
        stack_buf = _Buf()
        print("FASTHARNESS accumulate_slice prefetch=%d calib_out=%s "
              "fast_accum=1 lazy_nvalid=1 reused_stack=1"
              % (FAST_PREFETCH, use_out), flush=True)

        batches = [([int(k) for k in ks[i:i + subbatch]], bins[i:i + subbatch])
                   for i in range(0, len(ks), subbatch)]

        # ---- H1: prefetch.  read_events() returns a LIST, so the reader thread
        # genuinely materialises the next batch (pread + dgram assembly) while
        # the main thread calibrates the current one.  A FIFO queue of depth
        # FAST_PREFETCH preserves batch order EXACTLY, and ``ridx`` is touched by
        # this thread and no other.
        def _read_ahead():
            try:
                for sub, sub_bins in batches:
                    if stop.is_set():
                        break
                    q.put((ridx.read_events(sub), sub_bins))
            except BaseException as exc:                       # noqa: BLE001
                try:
                    q.put(exc)
                except Exception:                              # noqa: BLE001
                    pass
                return
            try:
                q.put(None)
            except Exception:                                  # noqa: BLE001
                pass

        reader = threading.Thread(target=_read_ahead, name="psdata-prefetch",
                                  daemon=True)
        reader.start()

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            evts, sub_bins = item
            for evt, b in zip(evts, sub_bins):
                b = int(b)
                if b < 0:
                    unbinned += 1
                    continue
                # H6: same algorithm as evt.stack(det), reused output buffer.
                raw = _stack_into(evt, det, stack_buf)
                if raw is None:
                    continue                    # detector absent from this event
                # ---- THE REAL CALIBRATION (this is what the old cube omitted)
                if use_out:
                    cal = pscalib.calib(dettype, raw, constants, config=seg_cfg,
                                        out=cal_buf.get(raw.shape, np.float32))
                else:
                    cal = pscalib.calib(dettype, raw, constants, config=seg_cfg)
                # H3: the finite test is REAL, per event -- nothing is assumed
                # about the constants.  Only an event that is genuinely all
                # finite takes the fast lane.
                finite = np.isfinite(cal, out=fin_buf.get(cal.shape, np.bool_))
                nfin = int(np.count_nonzero(finite))
                n_nan += cal.size - nfin
                if nfin == cal.size:
                    # np.where(all-True, cal, 0.0) IS cal, element for element
                    # (float32 preserved: the python 0.0 is a WEAK scalar in
                    # numpy 1.26.4), and nvalid[b] += all-True IS nvalid[b] += 1.
                    if b not in counts:
                        sums[b] = cal.astype(np.float64)
                        nfin_scalar[b] = 1
                        counts[b] = 1
                    else:
                        sums[b] += cal
                        nfin_scalar[b] = nfin_scalar.get(b, 0) + 1
                        counts[b] += 1
                else:
                    # VERBATIM the original expression.  Not one bit of this
                    # lane is rearranged.
                    contrib = np.where(finite, cal, 0.0)
                    if b not in counts:
                        sums[b] = contrib.astype(np.float64)
                        nvalid[b] = finite.astype(np.uint32)
                        nfin_scalar[b] = 0
                        counts[b] = 1
                    else:
                        sums[b] += contrib
                        if b in nvalid:
                            nvalid[b] += finite
                        else:
                            nvalid[b] = finite.astype(np.uint32)
                        counts[b] += 1
            # evts + their dgram snapshots drop out of scope here
        # ---- H4: materialise the lazy nvalid maps.  uint32 addition is exact
        # and associative, so this is bit-identical to having incremented the
        # map on every event.
        for b in counts:
            n = nfin_scalar.get(b, 0)
            if b in nvalid:
                if n:
                    nvalid[b] = nvalid[b] + np.uint32(n)
            else:
                nvalid[b] = np.full(sums[b].shape, n, dtype=np.uint32)
        # rchar is per-PROCESS, so the read bytes MUST be measured here, inside
        # the worker -- the driver's own /proc/self/io cannot see a Ray worker's
        # reads at all (separate processes), so a driver-side rchar delta would
        # report ~0 MB/s for every parallel run.  The prefetch thread reads in
        # THIS process, so its bytes are counted here too.
        return sums, nvalid, counts, n_nan, unbinned, rchar() - r0
    finally:
        stop.set()
        if reader is not None:
            # Drain so a reader blocked on a full queue can observe ``stop``.
            while reader.is_alive():
                try:
                    q.get_nowait()
                except Exception:                              # noqa: BLE001
                    reader.join(timeout=0.25)
            reader.join(timeout=30.0)
        ridx.close()


def _accumulate_slice_orig(ridx, r0, ks, bins, det, dettype, constants,
                           seg_cfg, subbatch, pscalib):
    """The UNMODIFIED loop, kept so BENCH_FAST_ACCUM=0 reproduces the original
    harness's behaviour inside this same file (development A/B only -- the proof
    job's after_orig arm loads the real, byte-identical original file)."""
    try:
        sums, nvalid, counts = {}, {}, {}
        n_nan = 0
        unbinned = 0
        for i in range(0, len(ks), subbatch):
            sub = [int(k) for k in ks[i:i + subbatch]]
            sub_bins = bins[i:i + subbatch]
            for evt, b in zip(ridx.read_events(sub), sub_bins):
                b = int(b)
                if b < 0:
                    unbinned += 1
                    continue
                raw = evt.stack(det)
                if raw is None:
                    continue
                cal = pscalib.calib(dettype, raw, constants, config=seg_cfg)
                finite = np.isfinite(cal)
                n_nan += int(cal.size - finite.sum())
                contrib = np.where(finite, cal, 0.0)
                if b not in sums:
                    sums[b] = contrib.astype(np.float64)
                    nvalid[b] = finite.astype(np.uint32)
                    counts[b] = 1
                else:
                    sums[b] += contrib
                    nvalid[b] += finite
                    counts[b] += 1
        return sums, nvalid, counts, n_nan, unbinned, rchar() - r0
    finally:
        ridx.close()
'''


# ==========================================================================
# LEVEL 2 (H7): the batched float64 accumulate.  Applied ON TOP of level 1 as
# three pure INSERTIONS -- every anchor below is reproduced verbatim inside its
# replacement, so not one level-1 character is deleted or rewritten.  main()
# asserts that (the level1 -> level2 diff must contain no '-' lines).
# ==========================================================================

# --------------------------------------------------------------------------
# (B1) the two env knobs, inserted straight after FAST_ACCUM
# --------------------------------------------------------------------------
OLD_B1 = '''FAST_ACCUM = int(os.environ.get("BENCH_FAST_ACCUM", "1"))
'''

NEW_B1 = '''FAST_ACCUM = int(os.environ.get("BENCH_FAST_ACCUM", "1"))
#: LEVEL 2 (H7).  Number of calibrated events buffered as float32 before they
#: are folded into the float64 accumulators in one grouped, slabbed pass.
#: 0 == OFF == exactly the level-1 code path, which is the DEFAULT: with this
#: knob unset bench_cube_calib_fast2.py behaves as bench_cube_calib_fast.py.
FAST_BATCH = int(os.environ.get("BENCH_FAST_BATCH", "0"))
#: Leading-axis (segment) entries per slab in the batched flush.  4 segments is
#: 8.4 MB of float32 against 16.8 MB of float64 -- L2/L3 resident.  Correctness
#: is independent of this value; only cache behaviour is not.
FAST_SLAB = int(os.environ.get("BENCH_FAST_SLAB", "4"))
'''

# --------------------------------------------------------------------------
# (B2) the dispatch, inserted between the FAST_ACCUM escape hatch and the
#      level-1 prefetch setup
# --------------------------------------------------------------------------
OLD_B2 = '''    if not FAST_ACCUM:
        return _accumulate_slice_orig(ridx, r0, ks, bins, det, dettype,
                                      constants, seg_cfg, subbatch, pscalib)
    reader = None
'''

NEW_B2 = '''    if not FAST_ACCUM:
        return _accumulate_slice_orig(ridx, r0, ks, bins, det, dettype,
                                      constants, seg_cfg, subbatch, pscalib)
    if FAST_BATCH > 0:
        # LEVEL 2 (H7).  OFF unless BENCH_FAST_BATCH is set, so the level-1
        # path below is byte-for-byte the proven one.
        return _accumulate_slice_batched(ridx, r0, ks, bins, det, dettype,
                                         constants, seg_cfg, subbatch, pscalib)
    reader = None
'''

# --------------------------------------------------------------------------
# (B3) the batched accumulator itself, appended after _accumulate_slice_orig
# --------------------------------------------------------------------------
OLD_B3 = '''        return sums, nvalid, counts, n_nan, unbinned, rchar() - r0
    finally:
        ridx.close()
'''

NEW_B3 = '''        return sums, nvalid, counts, n_nan, unbinned, rchar() - r0
    finally:
        ridx.close()


def _accumulate_slice_batched(ridx, r0, ks, bins, det, dettype, constants,
                              seg_cfg, subbatch, pscalib):
    """H7 -- LEVEL 2: the level-1 fast loop with the float64 add BATCHED.

    Reached only when ``BENCH_FAST_BATCH`` (``FAST_BATCH``) is > 0.  Up to K
    calibrated events are held as float32 contributions; the flush groups them
    by bin, PRESERVING ARRIVAL ORDER WITHIN EACH GROUP, and walks slabs of the
    bin's float64 accumulator adding the group's contributions to each slab in
    order.  Per element the sequence of additions is unchanged and distinct bins
    are disjoint arrays, so the float64 sums are BIT-IDENTICAL -- see the H7
    paragraph of patch_harness.py for the full argument.

    Everything else is level 1 verbatim: the H1 prefetch thread, the H3
    all-finite fast path with its real per-event isfinite, the H4 lazy nvalid
    (still updated EAGERLY per event -- only the float64 add is deferred), and
    the H6 reused stack buffer.  H2's single reused calib output buffer CANNOT
    survive here, because K events are alive at once; it becomes a pool of K
    buffers with a free list (see below).
    """
    reader = None
    stop = threading.Event()
    q = queue.Queue(maxsize=max(1, FAST_PREFETCH))
    nbatch = max(1, FAST_BATCH)
    nslab = max(1, FAST_SLAB)
    try:
        sums, nvalid, counts = {}, {}, {}
        nfin_scalar = {}
        n_nan = 0
        unbinned = 0
        use_out = _calib_supports_out(pscalib)
        fin_buf = _Buf()
        stack_buf = _Buf()
        # ---- H7 buffer pool.  ``pool`` holds up to nbatch reusable float32
        # calib output arrays, allocated LAZILY; ``free`` lists the slots whose
        # contents nothing points at.  A slot stays held while a buffered
        # contribution points into it and is handed back by the flush.
        pool = [None] * nbatch
        free = list(range(nbatch))
        pending = []            # [(bin, float32 contribution, slot or -1)]
        state = {"banner": False, "nalloc": 0, "flushes": 0, "maxlive": 0}
        print("FASTHARNESS accumulate_slice prefetch=%d calib_out=%s "
              "fast_accum=1 lazy_nvalid=1 reused_stack=1 batch=%d slab_seg=%d"
              % (FAST_PREFETCH, use_out, nbatch, nslab), flush=True)

        def _flush():
            """Fold every buffered contribution into its bin's accumulator.

            Grouped by bin (disjoint accumulators), arrival order preserved
            inside each group, slab by slab on the leading axis so each slab is
            a genuine view and stays cache-resident across the whole group.
            """
            if not pending:
                return
            state["flushes"] += 1
            if len(pending) > state["maxlive"]:
                state["maxlive"] = len(pending)
            order = []
            groups = {}
            for b, arr, _slot in pending:
                g = groups.get(b)
                if g is None:
                    g = groups[b] = []
                    order.append(b)
                g.append(arr)
            for b in order:
                g = groups[b]
                shape = g[0].shape
                nlead = shape[0]
                s = sums.get(b)
                first = 0
                if s is None:
                    # This bin's FIRST contribution initialises the float64
                    # accumulator: exactly ``g[0].astype(np.float64)``, written
                    # slab by slab over a partition that covers the array once.
                    s = np.empty(shape, dtype=np.float64)
                    sums[b] = s
                    first = 1
                for j0 in range(0, nlead, nslab):
                    j1 = j0 + nslab
                    if j1 > nlead:
                        j1 = nlead
                    sl = s[j0:j1]
                    if first:
                        sl[...] = g[0][j0:j1]
                    for c in g[first:]:
                        sl += c[j0:j1]
            for _b, _arr, slot in pending:
                if slot >= 0:
                    free.append(slot)
            del pending[:]

        batches = [([int(k) for k in ks[i:i + subbatch]], bins[i:i + subbatch])
                   for i in range(0, len(ks), subbatch)]

        # ---- H1: prefetch, verbatim the level-1 reader.
        def _read_ahead():
            try:
                for sub, sub_bins in batches:
                    if stop.is_set():
                        break
                    q.put((ridx.read_events(sub), sub_bins))
            except BaseException as exc:                       # noqa: BLE001
                try:
                    q.put(exc)
                except Exception:                              # noqa: BLE001
                    pass
                return
            try:
                q.put(None)
            except Exception:                                  # noqa: BLE001
                pass

        reader = threading.Thread(target=_read_ahead, name="psdata-prefetch",
                                  daemon=True)
        reader.start()

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            evts, sub_bins = item
            for evt, b in zip(evts, sub_bins):
                b = int(b)
                if b < 0:
                    unbinned += 1
                    continue
                # H6: same algorithm as evt.stack(det), reused output buffer.
                raw = _stack_into(evt, det, stack_buf)
                if raw is None:
                    continue                    # detector absent from this event
                # ---- THE REAL CALIBRATION, into a POOLED output buffer.
                if use_out and free:
                    slot = free.pop()
                    buf = pool[slot]
                    if (buf is None or buf.shape != raw.shape
                            or buf.dtype != np.float32):
                        buf = np.empty(raw.shape, dtype=np.float32)
                        pool[slot] = buf
                        state["nalloc"] += 1
                    cal = pscalib.calib(dettype, raw, constants, config=seg_cfg,
                                        out=buf)
                else:
                    # No out= support (or -- defensively -- an empty pool, which
                    # the len(pending) < nbatch invariant makes unreachable):
                    # calib allocates, and there is no slot to account for.
                    slot = -1
                    cal = pscalib.calib(dettype, raw, constants, config=seg_cfg)
                if not state["banner"]:
                    state["banner"] = True
                    fmb = cal.size * 4.0 / 1e6
                    pmb = (FAST_PREFETCH + 1) * subbatch * raw.nbytes / 1e6
                    print("FASTHARNESS batch MEM frame_f32=%.1f MB "
                          "live_frames<=%d buffered<=%.2f GB "
                          "prefetch=(%d+1)x%dev x %.1f MB=%.2f GB "
                          "per_bin_accum=%.1f MB(f64)+%.1f MB(u32) "
                          "bound_ex_accum=%.2f GB"
                          % (fmb, nbatch + 1, (nbatch + 1) * fmb / 1e3,
                             FAST_PREFETCH, subbatch, raw.nbytes / 1e6,
                             pmb / 1e3, cal.size * 8.0 / 1e6,
                             cal.size * 4.0 / 1e6,
                             ((nbatch + 1) * fmb + pmb) / 1e3), flush=True)
                # H3: the finite test is REAL, per event.
                finite = np.isfinite(cal, out=fin_buf.get(cal.shape, np.bool_))
                nfin = int(np.count_nonzero(finite))
                n_nan += cal.size - nfin
                counts[b] = counts.get(b, 0) + 1
                if nfin == cal.size:
                    # np.where(all-True, cal, 0.0) IS cal, so the buffered
                    # contribution is cal itself; nvalid[b] += all-True is += 1.
                    contrib = cal
                    nfin_scalar[b] = nfin_scalar.get(b, 0) + 1
                else:
                    # VERBATIM the original expression.  np.where ALWAYS returns
                    # a fresh array, which is what makes the identity test below
                    # a sound release condition for the pool slot.
                    contrib = np.where(finite, cal, 0.0)
                    if b in nvalid:
                        nvalid[b] += finite
                    else:
                        nvalid[b] = finite.astype(np.uint32)
                if slot >= 0 and contrib is not cal:
                    # slow lane: nothing points into the slot any more.
                    free.append(slot)
                    slot = -1
                pending.append((b, contrib, slot))
                if len(pending) >= nbatch:
                    _flush()
            # evts + their dgram snapshots drop out of scope here
        # ---- H7: flush whatever is still buffered when the slice ends (this
        # also covers the early break on the prefetch sentinel).
        _flush()
        print("FASTHARNESS batch DONE flushes=%d max_buffered=%d "
              "pool_frames_allocated=%d leftover_pending=%d"
              % (state["flushes"], state["maxlive"], state["nalloc"],
                 len(pending)), flush=True)
        # ---- H4: materialise the lazy nvalid maps.
        for b in counts:
            n = nfin_scalar.get(b, 0)
            if b in nvalid:
                if n:
                    nvalid[b] = nvalid[b] + np.uint32(n)
            else:
                nvalid[b] = np.full(sums[b].shape, n, dtype=np.uint32)
        return sums, nvalid, counts, n_nan, unbinned, rchar() - r0
    finally:
        stop.set()
        if reader is not None:
            while reader.is_alive():
                try:
                    q.get_nowait()
                except Exception:                              # noqa: BLE001
                    reader.join(timeout=0.25)
            reader.join(timeout=30.0)
        ridx.close()
'''

def _is_line_subsequence(old, new):
    """Do all lines of ``old`` occur, verbatim and in order, inside ``new``?"""
    it = iter(new.splitlines(True))
    return all(any(x == ln for x in it) for ln in old.splitlines(True))


LEVEL2 = ((OLD_B1, NEW_B1, "batch_knobs"),
          (OLD_B2, NEW_B2, "batch_dispatch"),
          (OLD_B3, NEW_B3, "batch_function"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--expect-md5", default=ORIG_MD5)
    ap.add_argument("--level", type=int, default=1, choices=(1, 2),
                    help="1 = the proven fast harness (default, byte-identical "
                         "to previous runs of this script); 2 = level 1 PLUS "
                         "the H7 batched accumulate, gated on BENCH_FAST_BATCH "
                         "whose default is 0 = OFF.")
    a = ap.parse_args()

    src = open(a.src, "rb").read()
    got = hashlib.md5(src).hexdigest()
    print("HARNESS_SRC        %s" % a.src)
    print("HARNESS_SRC_MD5    %s" % got)
    print("HARNESS_EXPECT_MD5 %s" % a.expect_md5)
    if got != a.expect_md5:
        print("FATAL: the shared harness is not the file this patch was written "
              "against.  Refusing -- re-derive the anchors before patching.")
        return 2
    text = src.decode("utf-8")

    n = 0
    for old, new, tag in ((OLD_IMPORTS, NEW_IMPORTS, "imports"),
                          (OLD_BODY, NEW_BODY, "accumulate_slice")):
        c = text.count(old)
        print("ANCHOR %-18s occurrences=%d" % (tag, c))
        if c != 1:
            print("FATAL: anchor %r matched %d times, expected exactly 1" % (tag, c))
            return 3
        text = text.replace(old, new, 1)
        n += 1

    level1_text = text
    print("HARNESS_LEVEL      %d" % a.level)
    print("LEVEL1_MD5         %s"
          % hashlib.md5(level1_text.encode("utf-8")).hexdigest())
    if a.level >= 2:
        for old, new, tag in LEVEL2:
            c = text.count(old)
            print("ANCHOR %-18s occurrences=%d" % (tag, c))
            if c != 1:
                print("FATAL: anchor %r matched %d times, expected exactly 1"
                      % (tag, c))
                return 3
            # Every line of the anchor must reappear, verbatim and in order,
            # inside the replacement (the replacement may INTERLEAVE new lines
            # between them -- the dispatch does exactly that -- but it may not
            # drop or rewrite one).  The whole-file diff below re-checks this.
            if not _is_line_subsequence(old, new):
                print("FATAL: level-2 anchor %r is not reproduced line for "
                      "line inside its replacement -- that would DELETE proven "
                      "level-1 text" % (tag,))
                return 4
            text = text.replace(old, new, 1)
            n += 1
        # Level 2 must be a pure INSERTION on top of level 1: every line of the
        # proven level-1 file must still be there, in order.  A '-' line in this
        # diff means a level-1 line was deleted or rewritten -- refuse.
        removed = [ln for ln in difflib.unified_diff(
            level1_text.splitlines(True), text.splitlines(True), n=0)
            if ln.startswith("-") and not ln.startswith("---")]
        added = [ln for ln in difflib.unified_diff(
            level1_text.splitlines(True), text.splitlines(True), n=0)
            if ln.startswith("+") and not ln.startswith("+++")]
        print("LEVEL2_DIFF        added_lines=%d removed_lines=%d "
              "insertion_only=%s" % (len(added), len(removed), not removed))
        if removed:
            print("FATAL: level 2 removed level-1 lines:")
            for ln in removed[:20]:
                sys.stdout.write("   " + ln)
            return 5

    out = text.encode("utf-8")
    with open(a.dst, "wb") as f:
        f.write(out)
    print("HARNESS_FAST       %s" % a.dst)
    print("HARNESS_FAST_MD5   %s" % hashlib.md5(out).hexdigest())
    print("REPLACEMENTS       %d" % n)
    print("SRC_UNCHANGED      %s" % (hashlib.md5(open(a.src, 'rb').read()).hexdigest() == got))
    print("\n--- unified diff (original -> fast) ---")
    for line in difflib.unified_diff(src.decode().splitlines(True),
                                     text.splitlines(True),
                                     fromfile="bench_cube_calib.py",
                                     tofile=os.path.basename(a.dst)):
        sys.stdout.write(line)
    if a.level >= 2:
        print("\n--- unified diff (level 1 -> level 2, INSERTIONS ONLY) ---")
        for line in difflib.unified_diff(level1_text.splitlines(True),
                                         text.splitlines(True),
                                         fromfile="bench_cube_calib_fast.py",
                                         tofile=os.path.basename(a.dst)):
            sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
