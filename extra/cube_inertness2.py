"""CUBE_INERT2 -- prove the LEVEL-2 harness (bench_cube_calib_fast2.py, the H7
BATCHED float64 accumulate) is NUMERICALLY INERT against the UNMODIFIED harness.

This file EXTENDS cube_inertness.py rather than editing it: cube_inertness.py is
imported by path and its ``load_by_path`` / ``compare_cubes`` / ``gain_census``
/ ``pick_mixing_pixel`` / ``md5`` are reused verbatim, so the comparison here is
literally the same raw-bit-pattern comparison that produced cube_inertness.log.
cube_inertness.py is left BYTE-UNCHANGED (its md5 is printed at start and end)
because it is the artifact behind an already-recorded PASS.

What is different from cube_inertness.py:

  * the compared file is bench_cube_calib_fast2.py, re-imported ONCE PER K --
    ``FAST_BATCH`` is read from the environment at IMPORT time, so a fresh
    module object per K is the only honest way to test several K in one process;
  * K in {1, 4, 64}: K=1 flushes every event (degenerate, must still be exact),
    K=4 flushes many times inside a read batch, K=64 spans several 16-event read
    batches and leaves a PARTIAL buffer for the end-of-slice flush;
  * FIVE phases, because the batched flush has more ways to be wrong than the
    level-1 loop:
      real           real constants, the harness's own bin rule.  Every event is
                     all-finite, so every buffered contribution is a POOL SLOT.
      nanlane        stage-1 and stage-2 pedestal planes poisoned to NaN.  Every
                     event is slow-lane, so every buffered contribution is a
                     FRESH np.where array and every pool slot is released early
                     -- the pool recycling path.
      mixlane        ONE pixel of the stage-0 pedestal plane poisoned, chosen by
                     cube_inertness.pick_mixing_pixel so that a single bin holds
                     BOTH lanes.  A flush group then mixes pool-slot arrays and
                     np.where arrays, and H4's ``nvalid[b] = partial + scalar``
                     merge is reached.
      interleave     real constants, bins forced to ROUND ROBIN
                     (0,1,2,3,0,1,2,3,...).  This is the case the grouping most
                     plausibly breaks: no bin's events are contiguous in arrival
                     order, so every flush group is assembled from scattered
                     positions and the arrival order inside the group is the
                     only thing keeping the sum bit-exact.
      interleave_nan round robin AND poisoned planes: interleaved groups whose
                     members are all fresh np.where arrays.
  * the arrival-order bin sequence actually used is PRINTED for every phase.
  * the batched lane is proved to have RUN, not assumed: the worker's own
    ``FASTHARNESS batch DONE flushes=... max_buffered=...`` line is captured
    from stdout, re-printed, and the flush count is checked against the count
    predicted from K and the number of cubed events.

Required output shape, one line per (phase, K):
    CUBE_INERT2 K=<k> bins=<n> sums_bitdiff=0 nvalid_diff=0 counts_equal=True
    n_nan=<a>/<b> PASS=True phase=<name>

Usage:
    python cube_inertness2.py [--events 96] [--bins 4] [--ks 1,4,64]
                              [--slab 4] [--bin-var epics:...]
Exit 0 PASS, 95 FAIL (cubes differ), 96 setup/coverage problem.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import re
import sys
import time

# Same reason as cube_inertness.py:75 -- the harness splits BENCH_WORKERS on
# comma AT IMPORT TIME, so an inherited '1 16' kills the import.
os.environ.pop("BENCH_WORKERS", None)

CAND2 = "/sdf/data/lcls/ds/prj/prjcwang31/results/_cand2"
CI = os.path.join(CAND2, "cube_inertness.py")
ORIG = ("/sdf/data/lcls/ds/prj/prjcwang31/results/_bench_runs/harness/"
        "bench_cube_calib.py")
FAST1 = os.path.join(CAND2, "bench_cube_calib_fast.py")
FAST2 = os.path.join(CAND2, "bench_cube_calib_fast2.py")
FIX = "/sdf/data/lcls/ds/prj/prjcwang31/results/_cand/fixture"
NPZ = os.path.join(CAND2, "cube_constants_mfx100848724_r51_jungfrau.npz")

DONE_RE = re.compile(r"FASTHARNESS batch DONE flushes=(\d+) max_buffered=(\d+) "
                     r"pool_frames_allocated=(\d+) leftover_pending=(\d+)")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def seq_str(bins, limit=200):
    s = ",".join(str(int(b)) for b in bins[:limit])
    return s + ("" if len(bins) <= limit else ",...(%d more)" % (len(bins) - limit))


def runs_of(bins):
    """Maximal contiguous blocks of one bin value -- runs > distinct bins means
    at least one bin's events are NOT contiguous in arrival order."""
    r = 0
    prev = object()
    for b in bins:
        if b != prev:
            r += 1
            prev = b
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=96)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--ks", default="1,4,64")
    ap.add_argument("--slab", type=int, default=4)
    ap.add_argument("--bin-var", dest="bin_var",
                    default="epics:pulse_energy_diode_txi")
    ap.add_argument("--det", default="jungfrau")
    a = ap.parse_args()
    Ks = [int(x) for x in a.ks.split(",") if x.strip()]

    import numpy as np

    ci = _load("cube_inertness_lib", CI)
    orig = ci.load_by_path("harness_orig", ORIG)
    import pscalib
    import psdata

    ci_md5_before = ci.md5(CI)
    print("CUBE_INERT2 SETUP orig=%s md5=%s" % (ORIG, ci.md5(ORIG)))
    print("CUBE_INERT2 SETUP fast1=%s md5=%s (NOT USED HERE, md5 witnessed only)"
          % (FAST1, ci.md5(FAST1)))
    print("CUBE_INERT2 SETUP fast2=%s md5=%s" % (FAST2, ci.md5(FAST2)))
    print("CUBE_INERT2 SETUP lib=%s md5=%s" % (CI, ci_md5_before))
    print("CUBE_INERT2 SETUP pscalib=%s" % pscalib.__file__)
    print("CUBE_INERT2 SETUP psdata=%s" % psdata.__file__)
    print("CUBE_INERT2 SETUP python=%s numpy=%s host=%s Ks=%s slab=%d"
          % (sys.version.split()[0], np.__version__, os.uname().nodename,
             Ks, a.slab), flush=True)

    spec = orig.DETECTORS[a.det]

    t0 = time.monotonic()
    r = psdata.open(exp=spec["exp"], run=spec["run"], dir=orig.DIR)
    ridx = r.build_index()
    n_total = ridx.n_events
    ts_all = np.asarray(ridx.timestamps, dtype=np.uint64)
    n = min(a.events, n_total)
    ks = sorted(set(int(x) for x in np.linspace(0, n_total - 1, n)))
    event_ts = ts_all[ks]
    print("CUBE_INERT2 INDEX %d events in run, cubing %d, index built %.2fs"
          % (n_total, len(ks), time.monotonic() - t0), flush=True)

    rule = orig.resolve_bin_rule(r, event_ts, a.bins, forced=a.bin_var)
    bins_real = [rule.bin_of(int(t)) for t in event_ts]
    hist = {}
    for b in bins_real:
        hist[b] = hist.get(b, 0) + 1
    n_unbinned = hist.pop(-1, 0)
    print("CUBE_INERT2 BINRULE label=%s tier=%s var=%r pv=%r nbins=%d "
          "fallback=%s" % (rule.label, rule.tier, rule.var, rule.pv,
                           rule.nbins, rule.is_fallback))
    print("CUBE_INERT2 BINS driver_side_occupancy=%s unbinned=%d"
          % ({k: hist[k] for k in sorted(hist)}, n_unbinned), flush=True)
    multi = [k for k in hist if hist[k] >= 2]
    if len(hist) < 2 or not multi:
        print("CUBE_INERT2 SETUP FAIL: degenerate coverage -- need >1 populated "
              "bin AND >=1 bin with several events; got %s"
              % ({k: hist[k] for k in sorted(hist)},))
        r.close()
        return 96

    # ---- the interleaved bin assignment: round robin over a.bins ------------
    nb = max(2, a.bins)
    bins_rr = [i % nb for i in range(len(ks))]

    print("\nCUBE_INERT2 ARRIVAL_ORDER phase=real n=%d distinct=%d runs=%d "
          "interleaved=%s" % (len(bins_real), len(set(bins_real)),
                              runs_of(bins_real),
                              runs_of(bins_real) > len(set(bins_real))))
    print("CUBE_INERT2 ARRIVAL_SEQ  phase=real %s" % seq_str(bins_real))
    print("CUBE_INERT2 ARRIVAL_ORDER phase=interleave n=%d distinct=%d runs=%d "
          "interleaved=%s" % (len(bins_rr), len(set(bins_rr)),
                              runs_of(bins_rr),
                              runs_of(bins_rr) > len(set(bins_rr))))
    print("CUBE_INERT2 ARRIVAL_SEQ  phase=interleave %s" % seq_str(bins_rr),
          flush=True)

    try:
        constants, cprov = orig.fetch_constants(r, spec, npz_path=NPZ)
    except Exception as e:                                     # noqa: BLE001
        print("CUBE_INERT2 CONSTANTS webdb/npz path FAILED (%s: %s) -- falling "
              "back to the frozen fixture constants (BOTH arms get the SAME "
              "dict, so the inertness comparison is unaffected)"
              % (type(e).__name__, e))
        constants = {}
        for ct in ("pedestals", "pixel_gain", "pixel_offset", "pixel_status"):
            p = os.path.join(FIX, ct + ".npy")
            if os.path.exists(p):
                constants[ct] = np.load(p)
        cprov = "fixture:%s ctypes=%s" % (FIX, sorted(constants))
    print("CUBE_INERT2 CONSTANTS %s" % cprov)
    seg_cfg = r.seg_configs(spec["det"]) if spec["needs_config"] else None
    state = ridx.to_dict()

    # ---- the poisoned constant sets ----------------------------------------
    ped = np.array(constants["pedestals"], copy=True)
    ped[1] = np.nan
    ped[2] = np.nan
    cons_nan = dict(constants)
    cons_nan["pedestals"] = ped

    t0 = time.monotonic()
    evbins, has13, iscode0 = ci.gain_census(state, ks, bins_real, spec["det"])
    print("CUBE_INERT2 GAIN_CENSUS %.2fs events=%d (driver-derived from the raw "
          "gain bits, independent of every harness)"
          % (time.monotonic() - t0, len(evbins)), flush=True)
    pick = ci.pick_mixing_pixel(evbins, iscode0)
    cons_mix = None
    if pick is None:
        print("CUBE_INERT2 MIXPIXEL none found in the sampled window -- the "
              "mixlane phase will be SKIPPED (not silently passed)")
    else:
        mb, row, col, n_code0, n_other = pick
        print("CUBE_INERT2 MIXPIXEL bin=%d segment=0 row=%d col=%d "
              "events_at_code0=%d events_at_other_code=%d"
              % (mb, row, col, n_code0, n_other))
        pedm = np.array(constants["pedestals"], copy=True)
        pedm[0, 0, row, col] = np.nan
        cons_mix = dict(constants)
        cons_mix["pedestals"] = pedm
    sys.stdout.flush()

    phases = [("real", constants, bins_real),
              ("nanlane", cons_nan, bins_real)]
    if cons_mix is not None:
        phases.append(("mixlane", cons_mix, bins_real))
    phases.append(("interleave", constants, bins_rr))
    phases.append(("interleave_nan", cons_nan, bins_rr))

    results = []
    allok = True
    for pname, cons, pbins in phases:
        occ = {}
        for b in pbins:
            b = int(b)
            if b >= 0:
                occ[b] = occ.get(b, 0) + 1
        print("\n" + "=" * 74)
        print("=== CUBE_INERT2 phase=%s occupancy=%s runs=%d distinct=%d ==="
              % (pname, {k: occ[k] for k in sorted(occ)}, runs_of(pbins),
                 len(set(pbins))))
        print("CUBE_INERT2 ARRIVAL_SEQ phase=%s %s" % (pname, seq_str(pbins)),
              flush=True)

        t0 = time.monotonic()
        A = orig.accumulate_slice(state, ks, pbins, spec["det"], spec["dettype"],
                                  cons, seg_cfg)
        print("CUBE_INERT2 RAN orig phase=%s %.2fs bins=%s n_nan=%d unbinned=%d"
              % (pname, time.monotonic() - t0, sorted(A[2]), A[3], A[4]),
              flush=True)
        n_cubed = sum(A[2].values())
        if pname in ("nanlane", "mixlane", "interleave_nan") and A[3] == 0:
            print("CUBE_INERT2 SETUP FAIL phase=%s: n_nan==0, the poisoned "
                  "constants did not force the slow lane, so this phase "
                  "proved nothing" % pname)
            allok = False

        for K in Ks:
            os.environ["BENCH_FAST_BATCH"] = str(K)
            os.environ["BENCH_FAST_SLAB"] = str(a.slab)
            mod = _load("harness_fast2_%s_K%d" % (pname, K), FAST2)
            if getattr(mod, "FAST_BATCH", None) != K:
                print("CUBE_INERT2 SETUP FAIL: fast2 module FAST_BATCH=%r, "
                      "expected %d" % (getattr(mod, "FAST_BATCH", None), K))
                allok = False
                continue
            if mod.accumulate_slice is orig.accumulate_slice:
                print("CUBE_INERT2 SETUP FAIL: fast2 and orig share one "
                      "function object")
                allok = False
                continue
            cap = io.StringIO()
            t0 = time.monotonic()
            with contextlib.redirect_stdout(cap):
                B = mod.accumulate_slice(state, ks, pbins, spec["det"],
                                         spec["dettype"], cons, seg_cfg)
            dt = time.monotonic() - t0
            txt = cap.getvalue()
            for ln in txt.splitlines():
                print("   [fast2 phase=%s K=%d] %s" % (pname, K, ln))
            m = DONE_RE.search(txt)
            batch_ran = m is not None
            exp_flush = n_cubed // K + (1 if n_cubed % K else 0)
            if batch_ran:
                fl, mx, na, left = (int(m.group(1)), int(m.group(2)),
                                    int(m.group(3)), int(m.group(4)))
                print("CUBE_INERT2 BATCHPROOF phase=%s K=%d flushes=%d "
                      "expected_flushes=%d max_buffered=%d expected_max=%d "
                      "pool_frames_allocated=%d leftover_pending=%d "
                      "batch_lane_ran=True"
                      % (pname, K, fl, exp_flush, mx, min(K, n_cubed), na,
                         left))
                if fl != exp_flush or left != 0 or mx > K:
                    print("CUBE_INERT2 BATCHPROOF FAIL phase=%s K=%d -- flush "
                          "bookkeeping is not what K implies" % (pname, K))
                    allok = False
            else:
                print("CUBE_INERT2 BATCHPROOF phase=%s K=%d batch_lane_ran="
                      "False -- the batched accumulator did NOT run, this "
                      "comparison proves nothing" % (pname, K))
                allok = False
            print("CUBE_INERT2 RAN fast2 phase=%s K=%d %.2fs bins=%s n_nan=%d "
                  "unbinned=%d (LOGIN NODE -- NOT A TIMING)"
                  % (pname, K, dt, sorted(B[2]), B[3], B[4]), flush=True)

            ok, bit, nv, ce, nbins = ci.compare_cubes(
                pname, A, B, "CUBE_BIN2_%s_K%d" % (pname.upper(), K))
            ok = ok and batch_ran
            line = ("CUBE_INERT2 K=%d bins=%d sums_bitdiff=%d nvalid_diff=%d "
                    "counts_equal=%s n_nan=%d/%d PASS=%s phase=%s"
                    % (K, nbins, bit, nv, ce, A[3], B[3], ok, pname))
            print(line, flush=True)
            results.append(line)
            allok = allok and ok
            del B
        del A

    print("\n" + "=" * 74)
    print("CUBE_INERT2 SUMMARY (%d comparisons)" % len(results))
    for line in results:
        print(line)
    ci_md5_after = ci.md5(CI)
    print("CUBE_INERT2 LIB_UNCHANGED %s md5_before=%s md5_after=%s"
          % (ci_md5_before == ci_md5_after, ci_md5_before, ci_md5_after))
    print("CUBE_INERT2 FAST1_MD5 %s (untouched by this run)" % ci.md5(FAST1))
    print("CUBE_INERT2_OVERALL PASS=%s" % allok, flush=True)
    r.close()
    return 0 if allok else 95


if __name__ == "__main__":
    sys.exit(main())
