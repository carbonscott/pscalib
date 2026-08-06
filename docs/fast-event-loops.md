# Fast event loops

`pscalib.calib()` is roughly 40 ms/event on jungfrau. A straightforward event
loop wrapped around it costs about 100 ms/event — so **most of the time in a
naive pipeline is not in `pscalib` at all**, it is in the scaffolding: the read,
the segment stack, the `isfinite`/`where` pass, and the float64 accumulate.

This page documents five caller-side moves that recover that overhead. They are
not part of `pscalib` — they live in *your* loop — but one of them (`out=`) needs
library support, and `pscalib.calib()` now provides it.

All five were measured together end to end and are **numerically inert**: the
binned cube they produce is bit-identical to the cube the unmodified loop
produces. Each move below carries the argument for why it cannot move a bit,
because "it looked the same" is not the standard.

## What they are worth

Measured on one exclusive milano node at SLAC S3DF, jungfrau `exp=mfx100848724
run=51`, 512 events at 33.5 MB/event, `workers=1`, slurm job 34324141. Same
`pscalib`, same events, same node; the only difference is the loop around
`calib()`.

**The result.** Cost per event at `workers=1`, lower is better:

| Event loop | ms/event | Speedup |
|---|---:|---:|
| naive | 116.1 | 1.00× |
| **with these five moves** | **76.9** | **1.51×** |

**Where the time goes.** Same job, measured in-situ at 512 events, `nbins=10`:

| Stage | Naive (ms/event) | Patched (ms/event) | Which move |
|---|---:|---:|---|
| read | 8.16 | 9.10 | 1 — prefetch (`overlap_f` 0.0000 → 0.9692) |
| segment stack | 7.91 | 6.98 | 5 — reused buffer |
| `pscalib.calib()` | 42.17 | 40.24 | 2 — `out=` buffer |
| accumulate | 41.70 | 14.98 | 3 + 4 — fast path, deferred `nvalid` |
| **total** | **99.94** | **62.48** | |

Read the second table as indicative rather than load-bearing. The `read` row is
wall-clock inside the reader thread rather than time on the critical path — at
`overlap_f=0.9692` almost none of it is — and `accumulate` is a residual
(`compute − stack − calib`), not an independent timer. The 116.1 → 76.9
end-to-end figure in the first table is the measured one.

## 1. Overlap the read with a prefetch thread

The single largest win. The read releases the GIL (it is `os.pread` underneath),
so a plain `threading.Thread` genuinely overlaps it with numpy compute — no
multiprocessing, no `asyncio`.

```python
import queue, threading

def _reader(ridx, batches, q):
    for sub in batches:
        q.put(ridx.read_events(sub))   # returns a list, in requested order
    q.put(None)                        # sentinel

q = queue.Queue(maxsize=1)             # depth 1 == classic double buffering
threading.Thread(target=_reader, args=(ridx, batches, q), daemon=True).start()

while (batch := q.get()) is not None:
    for evt in batch:
        ...                            # compute batch i while i+1 is read
```

**Why it cannot move a bit:** the same events are consumed in the same order.
Only the reader thread ever touches the reader object.

**What to expect.** Overlap is bounded by `(n_batches − 1) / n_batches`, so it
is poor for a handful of batches and excellent for many: measured `0.7489` at 4
batches against a `0.7500` ceiling, and `0.9692` at 32 batches against `0.9688`.
If your measured overlap is far below that ceiling, the reader is the
bottleneck, not the GIL.

One deployment note: under Ray, a task declared without `num_cpus` keeps full
CPU affinity, so the reader thread and the compute thread land on different
cores. A task pinned to a single core gets nothing from this.

## 2. Reuse the calibration output buffer — `out=`

Every `calib()` call otherwise first-touches 67 MB of fresh pages.

```python
cal_buf = np.empty((32, 512, 1024), dtype=np.float32)
...
cal = pscalib.calib(raw, constants, out=cal_buf)
```

`out=` is threaded through the public surface and through both plugins. Detect
it rather than assume it, so the same loop runs against an older `pscalib`:

```python
import inspect
supports_out = "out" in inspect.signature(pscalib.calib).parameters
```

**Why it cannot move a bit:** `out=` changes where the result is written, never
what is computed. The gate for this covers all four combinations —
allocated-vs-reference, `out`-vs-reference, `out`-vs-allocated, and
view-vs-allocated — at `max|diff| = 0` with zero sign-of-zero disagreements.

## 3. An all-finite fast path, taken on measured evidence

The usual accumulate is `contrib = np.where(finite, cal, 0.0)`, which allocates
and copies 67 MB per event. When every pixel is finite — the common case — that
result is `cal`, element for element.

```python
np.isfinite(cal, out=finite_buf)
nfin = np.count_nonzero(finite_buf)

if nfin == cal.size:
    sums[b] += cal                       # fast lane
else:
    sums[b] += np.where(finite_buf, cal, 0.0)   # verbatim original
```

**Why it cannot move a bit:** the branch is taken on a *counted* fact about this
event, never on an assumption. When the count equals `cal.size` every lane of
the `where` selects `cal`, and numpy 1.26.4 keeps float32 because the Python
`0.0` is a weak scalar. Any event with even one non-finite pixel falls through
to the original expression, unchanged.

**The trap, and it is a real one.** The obvious-looking alternative

```python
np.add(sums[b], cal, out=sums[b], where=finite)   # NOT equivalent
```

is wrong. A lane holding `-0.0` that the original would increment by `+0.0`
becomes `+0.0`; with `where=` it stays `-0.0`. On jungfrau this is not
hypothetical — `pixel_gain` is negative for gain stages 1 and 2, so masked
pixels really do carry `-0.0`, a few hundred to a few thousand per frame.

## 4. Defer the `nvalid` map

`nvalid[b] += finite` with an all-True `finite` is `nvalid[b] += 1`, and it
touches 67 MB of uint32 twice per event to say so. Count all-finite events per
bin as a Python `int`, materialise the map once at the end:

```python
nvalid[b] = partial_map + all_finite_count[b]
```

**Why it cannot move a bit:** integer addition is associative and exact.

## 5. Reuse the segment stack buffer

Do what `Event.stack` does, into a buffer allocated once, using only the public
`evt.raw()`:

```python
def _stack_into(evt, det_name, buf):
    segs = evt.raw(det_name)
    for k, s in enumerate(sorted(segs)):
        buf[k] = segs[s]
    return buf
```

**Why it cannot move a bit:** same segment order, same dtype, same copies — only
the 33.5 MB allocation is reused. The reader package is not modified.

## Two things that were tried and did not work

**A batched float64 accumulate — rejected, 0.95×.** Buffer K calibrated events
as float32, group by bin, and walk cache-sized slabs of the accumulator, cutting
traffic from `K × 335 MB` to `K × 67 MB + bins × 268 MB`. It is provably
bit-identical (per-element addition order is unchanged; distinct bins are
disjoint arrays). It is also *slower*: **0.9523× at `workers=1` and 0.53× at
`workers=16`**, because holding K live float32 frames costs more in cache
pressure and allocation than the saved traffic is worth. Do not re-derive this.

**Slabbing via `reshape(-1)` — silent data loss.** Taking slabs on a
non-contiguous view and flattening with `t.reshape(-1)` returns a *copy*, so
`+=` writes into a temporary and every accumulated value is discarded with no
error. Take slabs on axis 0 so each slab is a genuine view, and assert
`C_CONTIGUOUS` if you flatten.

## Still on the table

The `isfinite` pass itself (~6–8 ms/event) can be deleted rather than optimised:
finiteness of the output is a property of the *constants*, which are fetched
once per run, not of the per-event data. Prove it once at constant-load time and
the per-event pass goes away. Specified but not built.

## Checklist

1. Prefetch thread, queue depth 1, many batches. Print your overlap fraction and
   compare it against `(n−1)/n`.
2. `pscalib.calib(..., out=buf)`, with `out=` support detected not assumed.
3. All-finite fast path gated on `np.count_nonzero`, with the original
   expression kept verbatim as the fallback. Never `where=` on the accumulate.
4. `nvalid` deferred to the end.
5. Stack into a reused buffer.
6. Verify the whole thing is inert: compare your output against the unpatched
   loop as **raw bit patterns**, not with a tolerance, over the same events —
   including at least one run that forces the non-finite lane.
