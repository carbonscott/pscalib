#!/usr/bin/env python3
"""Random-access / batch / index-integrity stress for psdata vs psana.

Checks the paths the single-event acceptance tests don't stress:
  1. read_event_at(k) at positions spread across the WHOLE run -- the first
     event, the LAST event, an evenly-spread linspace sample, and +/-1 around
     every bigdata chunk-roll boundary -- == the forward stream's k-th event ==
     psana's k-th event (byte-exact raw).  NOT a head prefix.
  2. timestamp round-trip: read_event(ts) returns the same array; an absent ts
     raises KeyError.
  3. batch read_events(ks) / read_stack(ks) == the per-position singles, incl.
     scrambled / descending / duplicate ks.
  4. index completeness: psdata forward count == index count == psana event
     count, AND the timestamp SETS are equal -- ASSERTED, fail-closed, not
     merely printed.

The position generation (:func:`gen_positions`) and chunk-roll boundary
discovery (:func:`chunk_boundary_positions`) are importable, psana-free helpers
so the whole-run coverage guarantee can be unit-tested without SLAC data (see
``tests/test_gate01_randaccess_coverage.py``).
"""
import sys

import numpy as np

DIR = "/sdf/data/lcls/ds/prj/public01/xtc"

# default number of evenly-spread probe positions across the whole run
NSAMPLE = 300


def gen_positions(n_events, n=NSAMPLE, boundaries=None):
    """Seek positions spread across the WHOLE run ``[0, n_events-1]`` -- NOT a
    head prefix.

    Always includes the FIRST event (``0``) and the LAST event
    (``n_events-1``), plus an evenly-spread ``linspace`` sample of ``n`` points
    across the full range, plus (when given) ``+/-1`` around every position in
    ``boundaries`` -- the chunk-roll hazards where a naive prefix index stops.
    Every position is clamped into range; the result is a sorted list of
    distinct positions.

    With the default ``boundaries=None`` and ``n_events >= n`` the result has
    exactly ``n`` positions (the ``linspace`` endpoints ARE ``0`` and
    ``n_events-1``), its maximum is exactly ``n_events-1``, and it spans the
    whole run (positions well above ``n_events/2`` are present).  This is the
    property GATE-01 requires: the coverage must REACH THE LAST EVENT, never a
    head prefix.
    """
    n_events = int(n_events)
    if n_events <= 0:
        return []
    if n_events == 1:
        return [0]
    n = max(2, min(int(n), n_events))
    pos = set(int(p) for p in np.linspace(0, n_events - 1, n).astype(np.int64))
    # hazard-targeted anchors: the FIRST and the LAST event of the whole run
    pos.add(0)
    pos.add(n_events - 1)
    if boundaries:
        for b in boundaries:
            for p in (int(b) - 1, int(b), int(b) + 1):
                if 0 <= p < n_events:
                    pos.add(p)
    return sorted(pos)


def chunk_boundary_positions(idx):
    """Best-effort list of event positions ``k`` at a bigdata **chunk roll** --
    where some stream's dgram first lands in a new chunk file (``c000`` ->
    ``c001`` ...) -- together with ``k-1``.  A roll is exactly where a naive
    prefix index would silently stop, so :func:`gen_positions` probes +/-1
    around each.

    Reads only ``idx.entries`` (``entries[k] = {stream: (chunk_path, off,
    size)}``); returns ``[]`` if the index does not expose it.
    """
    try:
        entries = idx.entries
    except Exception:  # noqa: BLE001 -- purely a best-effort hazard sweep
        return []
    positions = set()
    last_chunk = {}
    for k, entry in enumerate(entries):
        try:
            items = list(entry.items())
        except Exception:  # noqa: BLE001
            continue
        for stream, rec in items:
            chunk_path = rec[0] if isinstance(rec, (tuple, list)) else rec
            prev = last_chunk.get(stream)
            if prev is not None and chunk_path != prev:
                positions.add(k)
                if k - 1 >= 0:
                    positions.add(k - 1)
            last_chunk[stream] = chunk_path
    return sorted(positions)


def run(exp, runno, det, n_sample=NSAMPLE):
    print(f"==== {exp} r{runno} {det} ====")
    import psdata
    from psana import DataSource

    r = psdata.open(exp=exp, run=runno, dir=DIR)

    # Build the index first so we know the TRUE length of the WHOLE run; the
    # index defines the canonical event set (== forward stream == psana).
    idx = r.build_index()
    n_events = len(idx.timestamps)

    # Seek positions spread across the WHOLE run [0, n_events-1]: first, LAST,
    # an evenly-spread sample, and +/-1 around every chunk-roll boundary.
    boundaries = chunk_boundary_positions(idx)
    ks = gen_positions(n_events, n=n_sample, boundaries=boundaries)
    kset = set(ks)
    assert ks and ks[0] == 0 and ks[-1] == n_events - 1, (
        f"position coverage is not whole-run: ks span "
        f"[{ks[0] if ks else None} .. {ks[-1] if ks else None}] of {n_events}")
    print(f"  index has {n_events} L1 events; probing {len(ks)} positions "
          f"[{ks[0]} .. {ks[-1]}] ({len(boundaries)} chunk-roll hazards)")

    # Forward stream over the WHOLE run: record EVERY timestamp (for the count
    # + ts-SET asserts) and the raw ONLY at the probe positions.
    fwd_ts_all = []
    fwd_raw = {}
    for i, evt in enumerate(r.events()):
        fwd_ts_all.append(int(evt.timestamp))
        if i in kset:
            fwd_raw[i] = evt.stack(det)
    nfwd = len(fwd_ts_all)
    print(f"  forward streamed {nfwd} events")

    fails = []

    # (1) read_event_at(k) == forward k-th, across the WHOLE run
    for k in ks:
        a = r.read_event_at(k).stack(det)
        if not np.array_equal(a, fwd_raw[k]):
            fails.append(f"read_event_at({k}) != forward[{k}]")
    # (2) read_event(ts) round-trip at those same whole-run positions
    for k in ks:
        a = r.read_event(fwd_ts_all[k]).stack(det)
        if not np.array_equal(a, fwd_raw[k]):
            fails.append(f"read_event(ts[{k}]) != forward[{k}]")
    # absent ts -> KeyError
    try:
        r.read_event(1)  # ts=1 cannot exist
        fails.append("read_event(absent ts) did NOT raise")
    except KeyError:
        pass
    # (3) batch == singles, with scrambled/descending/duplicate ks
    bks = [ks[-1], ks[0], ks[len(ks) // 2], ks[0]]  # descending-ish + dup
    bevts = r.read_events(bks)
    for j, k in enumerate(bks):
        if not np.array_equal(bevts[j].stack(det), fwd_raw[k]):
            fails.append(f"read_events batch[{j}] (k={k}) != forward")
    stack = r.read_stack(bks, det)
    for j, k in enumerate(bks):
        if not np.array_equal(stack[j], fwd_raw[k]):
            fails.append(f"read_stack[{j}] (k={k}) != forward")

    # (4) psana cross-check over the WHOLE run: raw + ts at every probe
    # position, plus EVERY psana timestamp for the count + ts-SET asserts.
    ds = DataSource(exp=exp, run=runno, dir=DIR)
    prun = next(ds.runs())
    pdet = prun.Detector(det)
    psana_ts_all = []
    for i, evt in enumerate(prun.events()):
        ts = int(evt.timestamp)
        psana_ts_all.append(ts)
        if i in kset:
            if i < len(fwd_ts_all) and fwd_ts_all[i] != ts:
                fails.append(f"psana ts[{i}] != psdata ts[{i}]")
            praw = pdet.raw.raw(evt)
            if praw is not None and not np.array_equal(praw, fwd_raw[i]):
                fails.append(f"psana raw[{i}] != psdata raw[{i}]")
    pcount = len(psana_ts_all)
    print(f"  psana streamed {pcount} events")

    if fails:
        for f in fails[:10]:
            print(f"  FAIL: {f}")
        print(f"  VERDICT=FAIL ({len(fails)} issues)")
        return False

    # (4b) HARD, fail-closed completeness asserts over the WHOLE run.  These
    # were previously only PRINTED ("forward got .. index has .."), so a tail
    # divergence -- where this project's live bugs live -- passed unnoticed.
    # Event COUNT equality and timestamp-SET equality across forward / index /
    # psana now FAIL the gate on any mismatch.
    assert nfwd == n_events == pcount, (
        f"event COUNT mismatch: forward={nfwd}, index={n_events}, "
        f"psana={pcount}")
    assert set(fwd_ts_all) == set(psana_ts_all), (
        "timestamp SET mismatch forward vs psana ("
        f"forward-only={len(set(fwd_ts_all) - set(psana_ts_all))}, "
        f"psana-only={len(set(psana_ts_all) - set(fwd_ts_all))})")
    assert set(idx.timestamps) == set(fwd_ts_all), (
        "timestamp SET mismatch index vs forward stream")

    print(f"  VERDICT=PASS ({len(ks)} whole-run positions [{ks[0]}..{ks[-1]}], "
          f"batch, round-trip, count+ts-set ASSERTED)")
    return True


def main():
    ok = True
    ok &= run("mfx100848724", 51, "jungfrau", n_sample=300)
    ok &= run("ued1010667", 177, "epixquad", n_sample=400)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
