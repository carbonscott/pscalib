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

Emits per-mismatch diagnostics (capped by --maxshow) and one grep-friendly
RESULT line. Exit code is nonzero if ANY divergence/error was seen.
"""
import argparse
import sys
import time
import traceback

import numpy as np


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--dir', default='/sdf/data/lcls/ds/prj/public01/xtc')
    ap.add_argument('--det', required=True)
    ap.add_argument('--nevents', type=int, default=50,
                    help='number of (present) events to actually check')
    ap.add_argument('--stride', type=int, default=1,
                    help='check every Nth event in the stream')
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
        if c['checked'] >= args.nevents:
            break
        if i % args.stride:
            continue
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
