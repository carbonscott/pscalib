#!/usr/bin/env python3
"""Many-event byte-exact cross-check of psdata + pscalib against psana.

Ground truth = psana (production install, via psconda). For each psana event we
look the event up in psdata BY TIMESTAMP (exercising random access + parse) and
compare raw; then feed psana's own ``det.raw._calibconst`` into ``pscalib.calib``
(BYO path) and compare the apply math; optionally assemble + compare the image.

Run in the psana env with psdata+pscalib src on PYTHONPATH (run_tests.sh style):

    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    PYTHONPATH=.../software/pscalib/src:.../software/psdata/src \
        python stress_xcheck.py --exp mfx100848724 --run 51 --det jungfrau \
        --nevents 200 --checks raw,calib

COR-09: the cross-check positions used to be a HEAD PREFIX -- events 0..N-1 from
the FRONT (the ``--stride`` flag was never once passed, so every run decimated
nothing and stopped after the first ~100 events).  That is structurally blind to
every known live divergence in this project, because they are ALL positional and
LATE: the chunk roll at k=37,120 (STR-01), the step-2 config override (CAL-02),
the ragged tail at k=17,872 (FAIL-01).  The cross-check now samples the WHOLE run
-- first + LAST event, an evenly-spread linspace, and +/-1 around every BeginStep
and every chunk-roll boundary (:func:`gen_positions`).  ``--stride`` is corrected
so its decimation REACHES THE TAIL (``nevents x stride >= n_events``,
:func:`stride_positions`), and ``--positions`` lets a caller request the
hazard-targeted list or an arbitrary position set.

The position-generation helpers (:func:`gen_positions`, :func:`stride_positions`,
:func:`step_boundary_positions`, :func:`chunk_boundary_positions`,
:func:`resolve_positions`) are importable and psana-free so the whole-run
coverage guarantee can be unit-tested without SLAC data (see
``tests/test_cor09_whole_run_coverage.py``).

Emits per-mismatch diagnostics (capped by --maxshow) and one grep-friendly
RESULT line. Exit code is nonzero if ANY divergence/error was seen.
"""
import argparse
import sys
import time
import traceback

import numpy as np

# default number of evenly-spread probe positions across the WHOLE run
DEFAULT_NSAMPLE = 200


def unwrap(v):
    """psana _calibconst stores (value, meta) tuples; return the bare value."""
    if isinstance(v, (tuple, list)) and v and isinstance(v[0], (np.ndarray, str)):
        return v[0]
    return v


def eq_raw(a, b):
    """Byte-exact equality for RAW frames -- STRICT, no ``equal_nan``.

    Raw is integer detector data with no NaN, so equality must stay bare
    byte-exact: two frames are equal iff every element matches exactly. Do NOT
    add ``equal_nan`` here -- a stray NaN in a raw frame is a real divergence we
    want to catch, not mask.
    """
    return np.array_equal(a, b)


def eq_calib(a, b):
    """NaN-aware equality for CALIBRATED data.

    jungfrau (and other detectors) mask bad/undefined pixels to ``NaN`` in the
    calibrated output. A bare ``np.array_equal`` returns False whenever ANY NaN
    is present, because ``NaN != NaN`` -- so a byte-for-byte identical calib
    array (matching NaN positions included) would spuriously FAIL the gate even
    though psdata/pscalib and psana agree. ``equal_nan=True`` treats matching
    NaN positions as equal, so this gate agrees with ``bench_index.py``'s
    ``eq_calib`` gate on the same detector/run (that was the GATE-03 defect: two
    gates using contradictory predicates for the same comparison).
    """
    return np.array_equal(a, b, equal_nan=True)


# ---------------------------------------------------------------------------
# COR-09 position generation -- importable, psana-free (numpy only).
# ---------------------------------------------------------------------------
def gen_positions(n_events, n=DEFAULT_NSAMPLE,
                  step_boundaries=None, chunk_boundaries=None):
    """Cross-check positions spread across the WHOLE run ``[0, n_events-1]`` --
    NOT a head prefix.

    Always includes the FIRST event (``0``) and the LAST event
    (``n_events-1``), plus an evenly-spread ``linspace`` sample of ``n`` points
    across the full range, plus hazard-targeted ``+/-1`` around every position
    in ``step_boundaries`` (BeginStep -- where a step-2 config override such as
    CAL-02 first bites) and every position in ``chunk_boundaries`` (bigdata
    chunk roll -- STR-01 / FAIL-01, where a naive prefix index silently stops).
    Every position is clamped into range; the result is a sorted list of
    distinct positions.

    With ``n_events >= n`` the result's maximum is exactly ``n_events-1`` and it
    spans the whole run (positions well into the final quarter are present).
    This is the property COR-09 requires: the calib/image cross-check must REACH
    THE LAST EVENT and the late hazards, never a front-100 prefix.
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
    for group in (step_boundaries, chunk_boundaries):
        if group:
            for b in group:
                for p in (int(b) - 1, int(b), int(b) + 1):
                    if 0 <= p < n_events:
                        pos.add(p)
    return sorted(pos)


def stride_positions(n_events, nevents, stride):
    """Decimated positions that REACH THE TAIL (the GATE-04 refinement).

    A raw ``--stride S`` merely DECIMATES A PREFIX: coverage is
    ``0, S, 2S, ... (nevents-1)*S``, so the tail (``n_events-1``) is reached
    only if ``nevents * stride >= n_events``.  This bumps the EFFECTIVE stride to
    at least ``ceil(n_events / nevents)`` so ``nevents * stride_eff >= n_events``
    holds and the decimation spans the whole run, then force-appends the LAST
    event so the tail is always probed regardless of arithmetic.
    """
    n_events = int(n_events)
    if n_events <= 0:
        return []
    if n_events == 1:
        return [0]
    nevents = max(2, int(nevents))
    stride = max(1, int(stride))
    # fewest strided steps that still span [0, n_events-1] within nevents probes
    need = -(-n_events // nevents)          # ceil(n_events / nevents)
    eff = max(stride, need)                 # nevents * eff >= n_events
    pos = set(range(0, n_events, eff))
    pos.add(0)
    pos.add(n_events - 1)                    # guarantee the tail is reached
    return sorted(pos)


def step_boundary_positions(l1_timestamps, step_timestamps):
    """L1Accept positions ``k`` at a **BeginStep** (step boundary): the first
    L1Accept event at/after each BeginStep transition timestamp.

    A step boundary is exactly where a per-step config override (CAL-02) first
    takes effect, so :func:`gen_positions` probes ``+/-1`` around each.  Pure
    numpy, psana-free: ``l1_timestamps`` is the run's ascending L1Accept ts array
    (``idx.timestamps``) and ``step_timestamps`` the scan store's BeginStep ts
    (``run.env_store("scan").timestamps()``).
    """
    if l1_timestamps is None or step_timestamps is None:
        return []
    l1 = np.asarray(l1_timestamps, dtype=np.uint64)
    if l1.size == 0:
        return []
    out = set()
    for ts in np.asarray(step_timestamps, dtype=np.uint64):
        k = int(np.searchsorted(l1, ts, side="left"))
        if 0 <= k < l1.size:
            out.add(k)
    return sorted(out)


def chunk_boundary_positions(idx):
    """Best-effort list of event positions ``k`` at a bigdata **chunk roll** --
    where some stream's dgram first lands in a new chunk file (``c000`` ->
    ``c001`` ...) -- together with ``k-1``.  A roll is exactly where a naive
    prefix index would silently stop (STR-01 / FAIL-01), so
    :func:`gen_positions` probes ``+/-1`` around each.

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


def resolve_positions(n_events, nevents=DEFAULT_NSAMPLE, stride=1,
                      positions=None,
                      step_boundaries=None, chunk_boundaries=None):
    """Resolve the final sorted list of cross-check positions from the CLI flags.

    Precedence (all reach the tail, none is a head prefix):

    * ``positions`` -- an explicit request.  Either the string ``"hazards"``
      (the whole-run + BeginStep/chunk-roll hazard list) or an iterable / comma
      string of explicit integer positions (clamped into range, distinct,
      sorted).
    * ``stride > 1`` -- corrected decimation that reaches the tail
      (:func:`stride_positions`, ``nevents x stride >= n_events``).
    * default (no flags) -- the whole-run + hazard sample (:func:`gen_positions`).
    """
    if positions is not None and positions != "":
        if isinstance(positions, str) and positions.strip().lower() == "hazards":
            return gen_positions(n_events, n=nevents,
                                  step_boundaries=step_boundaries,
                                  chunk_boundaries=chunk_boundaries)
        return _parse_explicit_positions(positions, n_events)
    if stride and int(stride) > 1:
        return stride_positions(n_events, nevents, stride)
    return gen_positions(n_events, n=nevents,
                         step_boundaries=step_boundaries,
                         chunk_boundaries=chunk_boundaries)


def _parse_explicit_positions(positions, n_events):
    """Clamp an explicit position request (comma string or iterable of ints)
    into ``[0, n_events-1]``; distinct + sorted."""
    n_events = int(n_events)
    if isinstance(positions, str):
        toks = [t for t in positions.replace(",", " ").split() if t]
        raw = [int(t) for t in toks]
    else:
        raw = [int(t) for t in positions]
    out = set()
    for p in raw:
        if p < 0:                       # negative -> from the tail (python-style)
            p += n_events
        if 0 <= p < n_events:
            out.add(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--dir', default='/sdf/data/lcls/ds/prj/public01/xtc')
    ap.add_argument('--det', required=True)
    ap.add_argument('--nevents', type=int, default=DEFAULT_NSAMPLE,
                    help='number of evenly-spread probe positions across the '
                         'WHOLE run (first + last + hazards are always added)')
    ap.add_argument('--stride', type=int, default=1,
                    help='decimate the run every Nth event; the EFFECTIVE '
                         'stride is bumped so nevents*stride >= n_events (the '
                         'decimation reaches the tail, not just a prefix)')
    ap.add_argument('--positions', default=None,
                    help='explicit cross-check positions: "hazards" for the '
                         'whole-run + BeginStep/chunk-roll hazard list, or a '
                         'comma list of event indices (negatives count from the '
                         'end). Overrides --stride/--nevents sampling.')
    ap.add_argument('--checks', default='raw,calib',
                    help='comma list of raw,calib,image')
    ap.add_argument('--dettype', default=None,
                    help='override det_type passed to pscalib.calib')
    ap.add_argument('--maxshow', type=int, default=5)
    args = ap.parse_args()
    checks = set(c.strip() for c in args.checks.split(',') if c.strip())

    import psdata
    import pscalib
    from psana import DataSource

    ds = DataSource(exp=args.exp, run=args.run, dir=args.dir)
    psrun = next(ds.runs())
    det = psrun.Detector(args.det)

    constants = None
    try:
        constants = det.raw._calibconst
    except Exception as e:  # noqa: BLE001
        print(f"[warn] no _calibconst: {e}")
    dettype = args.dettype or getattr(det.raw, '_dettype', None)
    print(f"[info] dettype={dettype} "
          f"constants={'None' if constants is None else sorted(constants)}")

    pr = psdata.open(exp=args.exp, run=args.run, dir=args.dir)

    # Build the index first so we know the TRUE length of the WHOLE run and the
    # positions of the late hazards (chunk rolls, BeginSteps).  The index defines
    # the canonical event set (== forward stream == psana).
    idx = pr.build_index()
    n_events = len(idx.timestamps)

    # Late-hazard positions: +/-1 around each is probed by gen_positions.
    chunk_boundaries = chunk_boundary_positions(idx)
    try:
        step_ts = pr.env_store("scan").timestamps()
    except Exception as e:  # noqa: BLE001 -- best-effort hazard sweep
        print(f"[warn] scan store timestamps unavailable: {e}")
        step_ts = None
    step_boundaries = step_boundary_positions(idx.timestamps, step_ts)

    # WHOLE-RUN cross-check positions: first + LAST + evenly-spread + hazards.
    # NOT the front-100 prefix the buggy gate used (COR-09).
    positions = resolve_positions(
        n_events, nevents=args.nevents, stride=args.stride,
        positions=args.positions,
        step_boundaries=step_boundaries, chunk_boundaries=chunk_boundaries)
    want = set(positions)
    if not positions:
        print("[warn] no cross-check positions resolved; nothing to do")
        print(f"RESULT exp={args.exp} run={args.run} det={args.det} "
              f"dettype={dettype} checked=0 raw_mism=0 calib_mism=0 "
              f"image_mism=0 raw_none=0 notfound=0 err=0 secs=0.0 VERDICT=PASS")
        sys.exit(0)
    maxpos = max(want)
    print(f"[info] index has {n_events} L1 events; cross-checking "
          f"{len(positions)} WHOLE-RUN positions [{positions[0]}..{positions[-1]}] "
          f"({len(step_boundaries)} BeginStep, {len(chunk_boundaries)} chunk-roll "
          f"hazards)")

    # epix-family needs the per-segment Configure object
    config = None
    if dettype and 'epix' in str(dettype).lower():
        try:
            config = pr.seg_configs(args.det)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] seg_configs failed: {e}")

    # geometry index maps for the image check (once)
    ix = iy = None
    image_ok = 'image' in checks
    if image_ok:
        try:
            geo = unwrap(constants['geometry'])
            ix, iy = pscalib.pixel_coord_indexes_from_text(geo)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] image disabled (geometry): {e}")
            image_ok = False

    c = dict(checked=0, raw_mism=0, calib_mism=0, image_mism=0,
             raw_none=0, notfound=0, err=0)
    shown = 0
    t0 = time.time()

    def show(msg):
        nonlocal shown
        if shown < args.maxshow:
            print(msg)
            shown += 1

    for i, evt in enumerate(psrun.events()):
        if i > maxpos:
            break              # every requested position already visited
        if i not in want:
            continue           # cross-check ONLY the whole-run/hazard positions
        try:
            praw = det.raw.raw(evt)
            if praw is None:
                c['raw_none'] += 1
                continue
            ts = int(evt.timestamp)
            try:
                draw = pr.read_event(ts).stack(args.det)
            except Exception as e:  # noqa: BLE001
                c['notfound'] += 1
                show(f"[NOTFOUND] i={i} ts={ts}: {e!r}")
                continue
            c['checked'] += 1

            if 'raw' in checks:
                if draw is None or draw.shape != praw.shape \
                        or not eq_raw(draw, praw):
                    c['raw_mism'] += 1
                    d = 'None' if draw is None else f"shape {draw.shape} vs {praw.shape}"
                    show(f"[RAW MISMATCH] i={i} ts={ts} {d}")
                    continue  # calib/image meaningless on wrong raw

            need_calib = ('calib' in checks or image_ok) and constants is not None
            mcal = pcal = None
            if need_calib:
                pcal = det.raw.calib(evt)
                try:
                    if dettype:
                        mcal = pscalib.calib(dettype, draw, constants, config=config)
                    else:
                        mcal = pscalib.calib(draw, constants, config=config)
                except Exception as e:  # noqa: BLE001
                    c['err'] += 1
                    show(f"[CALIB ERR] i={i} ts={ts}: {e!r}")
                    traceback.print_exc()
                    continue

            if 'calib' in checks and constants is not None:
                if pcal is None or mcal is None or mcal.shape != pcal.shape \
                        or not eq_calib(mcal, pcal):
                    c['calib_mism'] += 1
                    if pcal is not None and mcal is not None and mcal.shape == pcal.shape:
                        diff = np.abs(mcal.astype('f8') - pcal.astype('f8'))
                        show(f"[CALIB MISMATCH] i={i} ts={ts} "
                             f"max|diff|={diff.max():.6g} "
                             f"nbad={int(np.count_nonzero(diff))}/{diff.size}")
                    else:
                        show(f"[CALIB MISMATCH] i={i} ts={ts} "
                             f"pcal={None if pcal is None else pcal.shape} "
                             f"mcal={None if mcal is None else mcal.shape}")

            if image_ok and mcal is not None:
                pimg = det.raw.image(evt)
                mimg = pscalib.assemble_image(mcal, ix, iy)
                if pimg is None or mimg is None or mimg.shape != pimg.shape \
                        or not np.array_equal(mimg, pimg):
                    c['image_mism'] += 1
                    if pimg is not None and mimg is not None and mimg.shape == pimg.shape:
                        diff = np.abs(mimg.astype('f8') - pimg.astype('f8'))
                        show(f"[IMAGE MISMATCH] i={i} ts={ts} "
                             f"max|diff|={diff.max():.6g}")
                    else:
                        show(f"[IMAGE MISMATCH] i={i} ts={ts} "
                             f"pimg={None if pimg is None else pimg.shape} "
                             f"mimg={None if mimg is None else mimg.shape}")
        except Exception as e:  # noqa: BLE001
            c['err'] += 1
            show(f"[ERR] i={i}: {e!r}")
            traceback.print_exc()

    dt = time.time() - t0
    bad = c['raw_mism'] + c['calib_mism'] + c['image_mism'] + c['err'] + c['notfound']
    print(f"RESULT exp={args.exp} run={args.run} det={args.det} dettype={dettype} "
          f"checked={c['checked']} raw_mism={c['raw_mism']} "
          f"calib_mism={c['calib_mism']} image_mism={c['image_mism']} "
          f"raw_none={c['raw_none']} notfound={c['notfound']} err={c['err']} "
          f"secs={dt:.1f} VERDICT={'FAIL' if bad else 'PASS'}")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
