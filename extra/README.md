# extra/ — archival, unmaintained

Everything in this directory is an **archive**. It is not part of the pscalib
package, it is **not maintained**, it is **not run by the test suite**, and
nothing in `src/pscalib/` imports it. It is kept because it is the only
surviving home for code that has no other owner.

These scripts **hardcode site-specific absolute paths** (`/sdf/...` at SLAC
S3DF), **assume SLURM** and an `sbatch`/`srun` installation, and assume a
psana conda release plus a benchmark harness that is **not in this repository
and not under version control anywhere**. They will not run outside that
environment and they are not intended to. Treat them as documentation of how a
measurement was performed, not as tooling you can invoke.

Nothing here is imported by `src/pscalib/`, exercised by `run_tests.sh`, or
collected by pytest. `pyproject.toml` scopes package discovery to `src/` and
carries no pytest configuration, and this directory carries no `__init__.py`, no
`test_*.py` and no `conftest.py`.

## Why it is here

The pscalib jungfrau fast calibration path (`src/pscalib/apply/_fastcalib.py`,
which arrives with the separate numpy-only calibration pull request and is not
on `main` as of this commit) was developed and measured against a **shared
benchmark harness that is not under version control** — `git rev-parse` in its
directory reports no repository. There is therefore no upstream project to
contribute the harness patch to. The patcher and the proof scripts are the only
durable record of how the end-to-end number was produced, so they are archived
here rather than being lost with the scratch directory they were written in.

## What each file is

| File | What it is |
| --- | --- |
| `patch_harness.py` | Produces a patched copy of the benchmark harness from the **unmodified** original by exact string replacement, asserting the source md5 first. It never edits the harness in place; it prints both md5s and a unified diff. Its default level applies **two** exact-string replacements — the import block and the body of `accumulate_slice` — carrying five changes that are each required to be numerically inert: read prefetch, a reused `calib` output buffer, an all-finite fast path, a deferred `nvalid` merge, and a reused stack buffer. An opt-in `--level 2` adds three further insertion-only replacements for a batched float64 accumulate, which is gated off by default. |
| `cube_inertness2.py` | Proves the batched-accumulate harness variant is numerically inert against the unmodified harness: both accumulate paths run in one process over the same events and their float64 sums are compared as **raw bit patterns**, not with a tolerance. |
| `insitu_probe.py` | Decomposes the harness's own `accumulate_slice()` into read / stack / calib / accumulate phases on real events, un-clocked, to attribute cost. It imports the harness by path and calls the real function — nothing is reimplemented. |
| `proof4arm.sbatch` | The SLURM job that produced the end-to-end measurement: four clocked arms (incumbent, candidate, candidate+patched harness, incumbent again as a control), an optional strictly fail-soft fifth arm for the batched harness variant that decides nothing, and an un-clocked in-situ profile — 512 events on one exclusive node. |

The **output** of `patch_harness.py` is deliberately **not** committed. It is a
derived copy of a file this repository does not own; committing it would fork
someone else's harness into this tree. The patcher is committed, its output is
not.

## Known gaps in this archive

* `cube_inertness2.py` imports `cube_inertness.py` **by path**, and that file is
  not archived here. As shipped, `cube_inertness2.py` documents a comparison; it
  cannot be executed as-is.
* Every path inside these files points at the original scratch directories,
  which are not part of this repository and may no longer exist.
* The harness these scripts patch and profile is not here and cannot be
  redistributed from here.

## What is maintained instead

The durable, tested, portable outcome of that work lives in the package and its
docs, not here. Two of the three arrive with the separate numpy-only
calibration pull request and are not on `main` as of this commit:

* `src/pscalib/apply/_fastcalib.py` — the fast calibration path itself.
* `docs/fast-event-loops.md` — the caller-side event-loop pattern the harness
  patch demonstrated, written up so it can be applied without this archive.
* `run_tests.sh` / `tests/` — the acceptance suite, which covers the package and
  deliberately does not cover this directory. This one is already on `main`.
