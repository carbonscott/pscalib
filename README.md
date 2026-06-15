# pscalib

A standalone, pure-Python LCLS-II **calibration** package — the calibration
sibling to the [`psdata`](../psdata) reader, exactly as `psdata` is the
framework-free sibling to psana's xtc2 reader.

`pscalib` *retrieves* calibration constants (a snapshot-to-disk provider, or — as
of US-001 — a `requests`-only web-DB client; **no psana framework, no MPI, no
BYOA**) and *applies* them in pure numpy, **byte-exact vs psana**:

```
(raw − pedestal[gain_range]) [× common_mode] × gain × mask  + geometry assembly
```

> **Status.** This README documents what US-000 delivers (the scaffold + the
> lifted, byte-exact Jungfrau calib/HDR engine). Later stories add the web-DB
> provider (US-001), validity enforcement (US-002), the epix config accessor
> (US-003, a psdata change) and the epix10ka apply plugin (US-004), and a unified
> public API + multi-detector gate (US-005). This README is expanded in US-005.

## Install / layout

A standalone uv/hatchling package mirroring `psdata`'s `src/` layout.

```
src/pscalib/
  __init__.py            # public surface; importing it pulls in only numpy
  _purity.py             # the shared import-purity rule (FORBIDDEN_MODULES)
  apply/jungfrau.py      # Jungfrau 3-gain HDR gain decode (== det.raw.calib)
  apply/epix10ka.py      # NEW (US-004)
  providers/snapshot.py  # capture (lazy psana) + numpy reload of constants
  providers/webdb.py     # NEW (US-001): requests-only web-DB client
  geometry.py            # geometry text -> per-pixel image index maps
  image.py               # pixel-array -> 2-D image remap (== det.raw.image)
  render.py              # HDRImager: raw -> calib -> image, fully offline
  model.py / registry.py # NEW (US-001..US-005)
```

- **Core dependency: `numpy` only**, plus the numpy-only `psdata` package (used
  for raw arrays and — from US-003 — the detector Configure object).
- **`web` extra** (`pip install pscalib[web]`): adds `requests` + `bson` for the
  US-001 web-DB retrieval provider.
- There is deliberately **no `[psana]` extra**: psana is the SLAC production
  conda build (sourced via `psconda.sh`), not a pip/uv-installable package, and
  is needed only to *regenerate ground truth* / take the one-time snapshot.

## Quickstart — offline calibrated HDR render (US-000)

```python
from pscalib import load_snapshot, HDRImager   # pure numpy, no psana

snap   = load_snapshot("snapshots/jungfrau_r0051")   # reload pinned constants
imager = HDRImager(snap)                             # offline render engine
calib, image = imager.render(raw_stack)             # numpy only
#   calib  (32,512,1024) f32  == det.raw.calib(evt)   (max|diff| == 0)
#   image  (4216,4432)  f32   == det.raw.image(evt)   (max|diff| == 0)
```

The one-time snapshot (the only psana-using step, a lazy import) is taken once in
the psana env:

```python
from pscalib.providers.snapshot import snapshot_calib
from pscalib.geometry import cache_pixel_indexes_for_snapshot
d = snapshot_calib(exp="mfx100848724", run=51,
                   dir="/sdf/data/lcls/ds/prj/public01/xtc",
                   detname="jungfrau", out_dir="snapshots")
cache_pixel_indexes_for_snapshot(d)   # one-time geometry index-map derivation
```

## Import purity

Importing `pscalib` and running any offline path (reload a snapshot, apply
constants, assemble an image) — or, from US-001, a web fetch — must not pull in
the framework. The forbidden set **extends** psdata's
(`('psana','mpi4py','h5py')`) with `dgram` + `pymongo`:

```python
import pscalib
pscalib.FORBIDDEN_MODULES        # ('psana','mpi4py','h5py','dgram','pymongo')
pscalib.assert_no_framework_imports()   # raises if any leaked into sys.modules
```

psana is permitted only inside two **lazy** function-body imports, both one-time
prep steps: the snapshot capture (`providers/snapshot.py`) and the geometry
index-map derivation (`geometry.py`). The web path may import `requests`/`bson`
but never `pymongo` (reads go over HTTP, never a Mongo socket).

## psdata relationship — depend + supersede (no duplicate-canonical hazard)

`pscalib` **depends on** the numpy-only `psdata` package; it does **not** depend
on psana. US-000 **lifts** psdata's already-byte-exact `calib/snapshot.py` +
`hdr/{jungfrau,geometry,image,render}.py` into `pscalib` as the **canonical
home**.

psdata's copies are **retained** (psdata is under active development — not
deleted), but to avoid two independently-editable copies, **psdata's `calib` and
`hdr` modules are a re-export shim of `pscalib`** (chosen option (a)). There is
exactly one implementation of each function/class object — drift is structurally
impossible, and `tests/test_no_drift_us000.py` proves it by *identity* (the
psdata symbols `is` the pscalib symbols). The shim requires `pscalib` to be
importable alongside `psdata`.

## Environment & tests

- Work on **sdfiana025**. Repo root:
  `/sdf/data/lcls/ds/prj/prjcwang31/results/software/pscalib`; the `psdata`
  sibling is at `…/software/psdata`.
- To regenerate ground truth / take a snapshot:
  `source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh`.
- Run the acceptance suite (puts both `pscalib` and `psdata` src/ on
  `PYTHONPATH`; psana resolves to the production env):

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
bash run_tests.sh
```

The offline import-purity checks run without psana; the byte-exact /
non-regression checks skip cleanly (with a message) when psana is not importable.

## Reference dataset

- **Jungfrau (HDR):** `exp=mfx100848724 run=51
  dir=/sdf/data/lcls/ds/prj/public01/xtc det='jungfrau'` — raw `(32,512,1024)u16`;
  calib `(32,512,1024)f32`; image `(4216,4432)f32`;
  pedestals/pixel_gain/pixel_offset `(3,32,512,1024)f32`.
