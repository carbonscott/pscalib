# .goal/ — development record for the jungfrau fast calibration work

This directory is a **historical record**, not configuration and not tooling.
Nothing in `src/pscalib/` reads it, and the test suite does not touch it. It is
committed because `main` carries the commits of two development efforts — the
numba implementation, whose pull request was closed unmerged, and the numpy-only
implementation that shipped — and these files are the honest provenance of how
that work was produced: what was assumed, what was measured, what was
disbelieved and re-measured, and which claims were verified first-hand versus
taken on report.

## What is here

Two completed efforts, each as a four-file set:

| Set | Files | What it produced |
| --- | --- | --- |
| `pscalib-calib-speedup` | `.json` (contract), `.ledger.json`, `.claims.json`, `.SETUP.md` | The first fast jungfrau calibration path, implemented with numba. Its pull request was closed unmerged, as superseded rather than rejected on merit — but its commits reached `main` inside the shipping branch's history, so head commit `5c432ec` is an ancestor of `main` and the numba implementation stays recoverable from history. |
| `pscalib-numpy-only-8x` | `.json` (contract), `.ledger.json`, `.claims.json`, `.SETUP.md` | The shipping implementation: the same speed with **numpy alone**, no numba and no new runtime dependency. This is what `src/pscalib/apply/_fastcalib.py` is; it was merged into `main` with its history preserved, which is how the numba effort's commits came along with it. |

Plus one live document:

| File | Role |
| --- | --- |
| `pscalib-pr-merge-cycle.SETUP.json` | The newest and most accurate access document: remote host, tools, path map, repo facts and known tool defects, re-verified first-hand. **It supersedes both `SETUP.md` files for any live remote fact.** |

## Reading order

Start with `pscalib-pr-merge-cycle.SETUP.json` for anything about the
environment. Read a campaign's `.json` for what it was required to prove, its
`.claims.json` for what was actually established and how strongly, and its
`.ledger.json` for the turn-by-turn record.

In a `.claims.json`, `provenance` is a lookup, not a confidence score:
`verified` means the evidence cites an artifact that was surfaced first-hand;
`inherited` means the evidence is a report of that artifact rather than the
artifact; `inferred` means reasoning only.

## Why these files are not portable, and why they were not made portable

The two `SETUP.md` files, and the older contracts, ledgers and claims files,
contain a site hostname, a site username and absolute site paths. That is
deliberate and it is left alone.

These files record what each effort **assumed and observed at the time**.
Rewriting them to look cleaner — swapping a real hostname for a placeholder,
or a real path for a variable — would falsify the one thing that makes them
worth shipping. A hostname in a ledger is not a configuration knob a reader
must edit; it is a record of where a number was measured. Nobody needs to edit
these files to use this repository, because nothing in this repository reads
them.

New portability work therefore went into exactly one file:
`pscalib-pr-merge-cycle.SETUP.json`. It confines every machine- and
site-dependent fact to the four values named in its own `retarget` block, and
`paths.project_root` is the only one any path depends on — every other path in
that file is relative to it. Retargeting to another user, project or site is
that single edit; the remaining three are the login host, the account name and
the absolute path of a third-party production build that sits outside the
project tree and cannot be derived from it.

For the same reason, a recursive grep for a hostname, a username or an absolute
site path over this directory does **not** come back empty, and should not be
expected to. Outside `pscalib-pr-merge-cycle.SETUP.json` and the two `SETUP.md`
files, the hits are in four frozen records — the two older contracts and the
first effort's ledger and claims — plus `pscalib-numpy-only-8x`'s ledger and
claims, whose only hits are compute-node names recording where a job ran. None
of them is a value a reader must edit.

## What is deliberately absent

The record of the merge cycle that is landing this work — its own contract,
ledger and claims — is **not** here. Those three files were still being written while
this directory was being assembled, so committing them would have committed a
half-written record. They live outside the repository, alongside the two
campaign directories that produced the two record sets above:

```
<workspace>/pscalib/campaigns/pscalib-calib-speedup/.goal/      pscalib-calib-speedup.{json,ledger.json,claims.json,SETUP.md}
<workspace>/pscalib/campaigns/pscalib-calib-speedup-b/.goal/    pscalib-numpy-only-8x.{json,ledger.json,claims.json,SETUP.md}
                                                                pscalib-pr-merge-cycle.{json,ledger.json,claims.json,SETUP.json}
```

Only `pscalib-pr-merge-cycle.SETUP.json` was copied in from that third set.
