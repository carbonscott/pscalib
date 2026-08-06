#!/usr/bin/env python
"""INSITU PHASE PROBE -- decompose the harness's OWN ``accumulate_slice()``
into read / stack / calib / accumulate, at nbins=10, on real events.

WHY THIS EXISTS
---------------
A skeptic QUARANTINED the campaign's accumulate cost: the 43.94 ms/event figure
was measured with ONE bin, while the harness runs ``--bins 10``, and in-situ
smokes put it at 49-72 ms.  Nothing in the four-arm clock decomposes the
number, so this probe runs AFTER the clock, un-clocked, and prints the split.

WHAT IS ACTUALLY RUN
--------------------
The harness module (unmodified OR patched -- ``--harness`` picks the file) is
imported by path and its REAL ``accumulate_slice(index_state, ks, bins, det,
dettype, constants, seg_cfg, subbatch)`` is called in THIS process.  Nothing is
reimplemented: the function under the timers is the same function the Ray
workers run during the clocked arms.  The event set, the bin rule and the
constants are built with the harness's OWN helpers (``resolve_bin_rule``,
``fetch_constants``) from the same 512-event evenly-spread sample the arms use,
then strided down to ``--n`` events so the bins still vary.

HOW THE SPLIT IS TAKEN (no edit to any harness file)
----------------------------------------------------
Four wrappers are installed around functions the harness calls, each recording
wall time and the calling thread:

  read   : ``psdata.index.RunIndex.read_events``   (in the patched harness this
           runs in the prefetch thread, so its time is NOT on the critical path)
  stack  : ``psdata.stream.Event.stack``  (unmodified harness) and/or the
           patched harness's module-level ``_stack_into``
  calib  : ``pscalib.calib``  -- wrapped with ``functools.wraps`` so that the
           patched harness's ``_calib_supports_out()`` signature probe still
           sees ``out`` in the parameters.  The probe ASSERTS that it does,
           before and after wrapping, because silently losing ``out=`` would
           change which code path is being profiled.
  qwait  : ``queue.Queue.get`` on the MAIN thread -- the patched harness's
           main-thread block on the prefetch queue.  Without this the block
           would be charged to the accumulate.

``accum`` is then the RESIDUAL of the whole call:

    accum = total - (read_on_main + qwait + stack + calib)

so it carries the accumulate block itself plus per-event loop overhead plus the
patched harness's end-of-slice ``nvalid`` materialisation.  Every component is
printed, so the residual can be re-derived by anyone who disagrees with the
attribution.

    overlap_f = (read_total + comp - total) / min(read_total, comp)
    comp      = total - read_on_main - qwait   (= stack + calib + accum)

For the unmodified harness the read is inline, so ``read_total == read_on_main``
and ``overlap_f`` is 0 BY CONSTRUCTION -- that is the expected, not an
interesting, result.  For the patched harness it is the fraction of the read
that the prefetch thread actually hid.

TWO SHORT-RUN ARTIFACTS, BOTH FOUND IN A LOGIN-NODE SMOKE OF THIS FILE AND BOTH
CORRECTED FOR HERE -- a 64-event probe is NOT automatically representative of a
512-event arm:

  (1) the gfm MEMO.  The candidate pscalib derives its folded gain/mask planes
      ONCE and memoises them.  In the smoke that one-time derivation cost about
      1.5 s, which over 16 events is +93 ms/event of pure startup charged to
      ``calib_ms`` (139.9 ms/event on the first pass vs 46.5 on the second pass
      over the same events).  Over the arm's 512 events the same 1.5 s is 2.9
      ms/event.  So an UNTIMED ``--warmup`` slice runs first and its wall is
      printed as INSITU_WARMUP; every reported line is therefore steady-state.
  (2) FIRST TOUCH.  With 10 bins, the first event in each bin pays a 134 MB
      ``astype(float64)`` + 67 MB uint32 allocation.  At n=64 that is 10/64 =
      16% of events; in the arm it is 10/512 = 2%.  ``n_first_touch`` is on
      every line, and the probe also runs n=256 and n=512 (the arm's own event
      count, so no extrapolation is needed at all) plus a one-bin rerun -- the
      one-bin case being exactly the configuration the quarantined 43.94 ms
      number was taken in.

Writes nothing.  Read-only against psdata / pscalib / the harness files.
"""
import argparse
import functools
import hashlib
import importlib.util
import os
import queue
import sys
import threading
import time

import numpy as np

# The harness reads BENCH_WORKERS at IMPORT time and splits it on COMMA; an
# inherited '1 16' would kill the import before anything else happens.
os.environ.pop("BENCH_WORKERS", None)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Tally(object):
    """Per-phase wall time.  Only two threads ever touch this and they touch
    DIFFERENT attributes (reader -> read, main -> everything else), so no lock
    is needed; ``read`` is incremented by whichever thread did the read."""

    def __init__(self):
        self.main_ident = threading.get_ident()
        self.read = 0.0
        self.read_main = 0.0
        self.stack = 0.0
        self.calib = 0.0
        self.qwait = 0.0
        self.n_read = 0
        self.n_stack = 0
        self.n_calib = 0
        self.n_qget = 0

    def reset(self):
        Tally.__init__(self)


T = Tally()


def install_wrappers(mod, pscalib):
    """Wrap read/stack/calib/queue-get.  Returns a restore() callable."""
    undo = []

    from psdata.index import RunIndex
    _read = RunIndex.read_events

    @functools.wraps(_read)
    def w_read(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return _read(self, *a, **kw)
        finally:
            dt = time.perf_counter() - t0
            T.read += dt
            T.n_read += 1
            if threading.get_ident() == T.main_ident:
                T.read_main += dt

    RunIndex.read_events = w_read
    undo.append(lambda: setattr(RunIndex, "read_events", _read))

    # -- stack: the unmodified harness calls evt.stack(det); the patched one
    #    calls the module-level _stack_into(evt, det, buf).  Wrap whichever
    #    exists -- wrapping both is harmless because only one is ever called.
    ev_cls = None
    try:
        import psdata.stream as _st
        ev_cls = getattr(_st, "Event", None)
    except Exception as exc:                                    # noqa: BLE001
        print("INSITU WARN cannot import psdata.stream: %r" % (exc,))
    if ev_cls is not None and hasattr(ev_cls, "stack"):
        _stack = ev_cls.stack

        @functools.wraps(_stack)
        def w_stack(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return _stack(self, *a, **kw)
            finally:
                T.stack += time.perf_counter() - t0
                T.n_stack += 1

        ev_cls.stack = w_stack
        undo.append(lambda: setattr(ev_cls, "stack", _stack))
        print("INSITU wrapped psdata.stream.Event.stack")
    else:
        print("INSITU WARN psdata.stream.Event.stack NOT found -- stack_ms will "
              "be 0 for the unmodified harness and the time falls into accum_ms")

    if hasattr(mod, "_stack_into"):
        _si = mod._stack_into

        @functools.wraps(_si)
        def w_si(*a, **kw):
            t0 = time.perf_counter()
            try:
                return _si(*a, **kw)
            finally:
                T.stack += time.perf_counter() - t0
                T.n_stack += 1

        mod._stack_into = w_si
        undo.append(lambda: setattr(mod, "_stack_into", _si))
        print("INSITU wrapped harness._stack_into")

    # -- calib
    _calib = pscalib.calib
    sig_before = _supports_out(pscalib)

    @functools.wraps(_calib)
    def w_calib(*a, **kw):
        t0 = time.perf_counter()
        try:
            return _calib(*a, **kw)
        finally:
            T.calib += time.perf_counter() - t0
            T.n_calib += 1

    pscalib.calib = w_calib
    undo.append(lambda: setattr(pscalib, "calib", _calib))
    sig_after = _supports_out(pscalib)
    print("INSITU wrapped pscalib.calib  supports_out before=%s after=%s"
          % (sig_before, sig_after))
    assert sig_before == sig_after, (
        "wrapping pscalib.calib CHANGED the detected out= support (%s -> %s); "
        "the patched harness would take a different code path under the probe "
        "than it does in the clocked arm, so the split would be a lie"
        % (sig_before, sig_after))

    # -- the main thread's block on the prefetch queue
    _get = queue.Queue.get

    @functools.wraps(_get)
    def w_get(self, *a, **kw):
        if threading.get_ident() != T.main_ident:
            return _get(self, *a, **kw)
        t0 = time.perf_counter()
        try:
            return _get(self, *a, **kw)
        finally:
            T.qwait += time.perf_counter() - t0
            T.n_qget += 1

    queue.Queue.get = w_get
    undo.append(lambda: setattr(queue.Queue, "get", _get))

    def restore():
        for f in reversed(undo):
            f()

    return restore


def _supports_out(pscalib):
    try:
        import inspect
        return "out" in inspect.signature(pscalib.calib).parameters
    except Exception:                                           # noqa: BLE001
        return False


def load_harness(path):
    spec = importlib.util.spec_from_file_location("bench_harness_insitu", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_harness_insitu"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(mod, pscalib, state, ks, bins, spec, constants, seg_cfg, label,
            nbins_tag):
    T.reset()
    t0 = time.perf_counter()
    out = mod.accumulate_slice(state, ks, bins, spec["det"], spec["dettype"],
                               constants, seg_cfg, mod.SUBBATCH)
    total = time.perf_counter() - t0
    sums, nvalid, counts, n_nan, unbinned, nbytes = out

    n = sum(counts.values())
    n = max(1, n)
    read = T.read / n * 1e3
    read_main = T.read_main / n * 1e3
    stack = T.stack / n * 1e3
    calib = T.calib / n * 1e3
    qwait = T.qwait / n * 1e3
    tot = total / n * 1e3
    accum = tot - read_main - qwait - stack - calib
    comp = tot - read_main - qwait
    denom = min(read, comp)
    f = (read + comp - tot) / denom if denom > 0 else float("nan")

    # The mandated key=value order comes FIRST and verbatim; the label is a
    # trailing key so a strict prefix parse of the line still works.
    print("INSITU nbins=%s n=%d read_ms=%.3f stack_ms=%.3f calib_ms=%.3f "
          "accum_ms=%.3f total_ms=%.3f overlap_f=%.4f harness=%s"
          % (nbins_tag, n, read, stack, calib, accum, tot, f, label),
          flush=True)
    print("INSITU_DETAIL nbins=%s n=%d wall_s=%.3f read_total_ms=%.3f "
          "read_on_main_ms=%.3f qwait_ms=%.3f comp_ms=%.3f n_batches=%d "
          "overlap_f_ceiling=%.4f n_bins_hit=%d n_first_touch=%d n_nan=%d "
          "unbinned=%d rchar_MB=%.1f n_read_calls=%d n_stack_calls=%d "
          "n_calib_calls=%d n_qget=%d harness=%s occupancy=%s"
          % (nbins_tag, n, total, read, read_main, qwait, comp,
             T.n_read, (1.0 - 1.0 / T.n_read) if T.n_read else float("nan"),
             len(counts), len(counts), n_nan, unbinned, nbytes / 1e6,
             T.n_read, T.n_stack, T.n_calib, T.n_qget, label,
             {k: counts[k] for k in sorted(counts)}), flush=True)
    return dict(n=n, read=read, stack=stack, calib=calib, accum=accum,
                total=tot, f=f, qwait=qwait, comp=comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--pool", type=int, default=512,
                    help="the arm's event set size; --n is a stride of it")
    ap.add_argument("--bin-var", default="epics:pulse_energy_diode_txi")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--also-one-bin", action="store_true")
    ap.add_argument("--extra-n", default="",
                    help="comma-separated extra event counts, e.g. 256,512")
    ap.add_argument("--warmup", type=int, default=4,
                    help="UNTIMED events run first so the reported lines are "
                         "steady-state: pscalib derives its gain/mask memo ONCE "
                         "(~1.5 s), which over 64 events would be +23 ms/event "
                         "of pure startup charged to calib_ms. 0 disables.")
    a = ap.parse_args()

    print("=" * 78, flush=True)
    print("INSITU PROBE label=%s host=%s slurm=%s pid=%d"
          % (a.label, os.uname()[1], os.environ.get("SLURM_JOB_ID"), os.getpid()))
    print("INSITU harness=%s md5=%s" % (a.harness, md5(a.harness)))
    mod = load_harness(a.harness)
    import psdata
    import pscalib
    print("INSITU psdata=%s" % psdata.__file__)
    print("INSITU pscalib=%s" % pscalib.__file__)
    print("INSITU SUBBATCH=%d FAST_ACCUM=%s FAST_PREFETCH=%s"
          % (mod.SUBBATCH, getattr(mod, "FAST_ACCUM", "n/a"),
             getattr(mod, "FAST_PREFETCH", "n/a")), flush=True)

    spec = mod.DETECTORS["jungfrau"]
    t0 = time.monotonic()
    r = psdata.open(exp=spec["exp"], run=spec["run"], dir=mod.DIR)
    ridx = r.build_index()
    print("INSITU index built n_events=%d in %.2fs"
          % (ridx.n_events, time.monotonic() - t0), flush=True)
    ts_all = np.asarray(ridx.timestamps, dtype=np.uint64)
    n_total = ridx.n_events

    # EXACTLY the arm's event set, then strided down.
    pool = min(a.pool, n_total)
    ks_pool = sorted(set(int(x) for x in np.linspace(0, n_total - 1, pool)))
    ts_pool = ts_all[ks_pool]
    rule = mod.resolve_bin_rule(r, ts_pool, a.nbins, forced=a.bin_var)
    bins_pool = [rule.bin_of(int(t)) for t in ts_pool]
    print("INSITU bin rule=%s nbins=%d pool=%d"
          % (rule.label, a.nbins, len(ks_pool)), flush=True)

    t0 = time.monotonic()
    constants, cprov = mod.fetch_constants(r, spec, npz_path=a.npz)
    print("INSITU constants %.2fs %s" % (time.monotonic() - t0, cprov),
          flush=True)
    seg_cfg = r.seg_configs(spec["det"]) if spec["needs_config"] else None
    state = ridx.to_dict()

    def subset(nev):
        step = max(1, len(ks_pool) // nev)
        idx = list(range(0, len(ks_pool), step))[:nev]
        return [ks_pool[i] for i in idx], [bins_pool[i] for i in idx]

    restore = install_wrappers(mod, pscalib)
    try:
        # ---- UNTIMED warmup: builds pscalib's one-time gain/mask memo so it is
        # not amortised over a 64-event probe (see the module docstring).
        if a.warmup > 0:
            wks, wbins = subset(a.warmup)
            t0 = time.perf_counter()
            mod.accumulate_slice(state, wks, wbins, spec["det"],
                                 spec["dettype"], constants, seg_cfg,
                                 mod.SUBBATCH)
            print("INSITU_WARMUP harness=%s n=%d wall_s=%.3f (UNTIMED; every "
                  "INSITU line below is steady-state)"
                  % (a.label, len(wks), time.perf_counter() - t0), flush=True)

        extras = [int(x) for x in a.extra_n.split(",") if x.strip()]
        first = True
        for nev in [a.n] + extras:
            ks, bins = subset(nev)
            run_one(mod, pscalib, state, ks, bins, spec, constants, seg_cfg,
                    a.label, a.nbins)
            if a.also_one_bin and first:
                run_one(mod, pscalib, state, ks, [0] * len(ks), spec,
                        constants, seg_cfg, a.label + "_ONEBIN", 1)
            first = False
    finally:
        restore()
        ridx.close()
        r.close()
    print("INSITU DONE label=%s" % a.label, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
