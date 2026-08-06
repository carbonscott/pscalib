# .goal/ — development record for the jungfrau fast calibration work

This directory is a **historical record**, not configuration and not tooling.
Nothing in `src/pscalib/` reads it, and the test suite does not touch it. It is
committed because this repository carries the work of two development efforts —
one closed without merging but recoverable from its branch, one still in review
at the time of this commit — and these files are the honest provenance of how
that work was produced: what
was assumed, what was measured, what was disbelieved and re-measured, and which
claims were verified first-hand versus taken on report.

## What is here

Two completed efforts, each as a four-file set:

| Set | Files | What it produced |
| --- | --- | --- |
| `pscalib-calib-speedup` | `.json` (contract), `.ledger.json`, `.claims.json`, `.SETUP.md` | The first fast jungfrau calibration path, implemented with numba. Its head commit is preserved on its own branch and recoverable from there; it is **not** in `main`'s history and was **not** merged as the shipping implementation. |
| `pscalib-numpy-only-8x` | `.json` (contract), `.ledger.json`, `.claims.json`, `.SETUP.md` | The shipping implementation: the same speed with **numpy alone**, no numba and no new runtime dependency. This is what `src/pscalib/apply/_fastcalib.py` is; it arrives with its own pull request and is not on `main` as of this commit. |

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
`pscalib-pr-merge-cycle.SETUP.json`, which resolves every machine- and
site-dependent fact through a single editable value, `paths.project_root`.
Retargeting it to another user, project or site is a one-line edit.

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
