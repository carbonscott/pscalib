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

**What it does, end to end.** Get a detector's calibration constants from *any*
provider — a snapshot on disk (US-000), the on-site web-DB over HTTP (US-001), or
your own dict — then calibrate raw frames in pure numpy with one call,
`pscalib.calib(raw, constants, config=None)`, byte-exact vs `det.raw.calib(evt)`
for both **jungfrau** and **epix10ka** (US-004). Staleness is enforced
refuse-by-default (US-002). The whole thing imports only numpy; `requests`+`bson`
come in only under the `web` extra; psana never (it is used only to regenerate
ground truth or take the one-time snapshot, via two lazy imports).

## Install / layout

A standalone uv/hatchling package mirroring `psdata`'s `src/` layout.

```
src/pscalib/
  __init__.py            # public surface; importing it pulls in only numpy
  _purity.py             # the shared import-purity rule (FORBIDDEN_MODULES)
  registry.py            # pscalib.calib(...) dispatch: det_type -> apply plugin
  model.py               # Constants contract + Pin + Validity + staleness check
  apply/jungfrau.py      # Jungfrau 3-gain calibration (== det.raw.calib)
  apply/epix10ka.py      # epix10ka 7-gain-range decode (config-driven)
  providers/snapshot.py  # capture (lazy psana) + numpy reload of constants
  providers/webdb.py     # requests-only web-DB client (the `web` extra)
  geometry.py            # geometry text -> per-pixel image index maps
  image.py               # pixel-array -> 2-D image remap (== det.raw.image)
  render.py              # Imager: raw -> calib -> image, fully offline
```

- **Core dependency: `numpy` only**, plus the numpy-only `psdata` package (used
  for raw arrays and — from US-003 — the detector Configure object).
- **`web` extra** (`pip install pscalib[web]`): adds `requests` + `bson` for the
  US-001 web-DB retrieval provider.
- There is deliberately **no `[psana]` extra**: psana is the SLAC production
  conda build (sourced via `psconda.sh`), not a pip/uv-installable package, and
  is needed only to *regenerate ground truth* / take the one-time snapshot.

## Quickstart — offline calibrated render (US-000)

```python
from pscalib import load_snapshot, Imager   # pure numpy, no psana

snap   = load_snapshot("snapshots/jungfrau_r0051")   # reload pinned constants
imager = Imager(snap)                               # offline render engine
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

## Public API

One call calibrates raw frames for any supported detector:

```python
import pscalib

# `constants` is any provider's output (see Providers below); `config` is the
# per-segment Configure object detectors like epix10ka need (jungfrau ignores it).
calib = pscalib.calib(raw, constants, config=None)
#   -> ndarray, byte-exact vs det.raw.calib(evt)
```

`pscalib.calib` **infers the detector type from the constants themselves** — a
snapshot's / web fetch's `detname`/`dettype` metadata, or a `dettype` key on a
BYO dict — and dispatches through the registry to the right pure-numpy apply
plugin. There is **no `det_type` argument** in this form. (An explicit form,
`pscalib.calib(det_type, raw, constants, config=None)`, is also accepted when you
want to pin the type yourself.)

### Validity enforcement (refuse-by-default)

Pass the run you are calibrating to enforce that the constants are valid for it
(US-002). Out-of-range **raises** by default:

```python
pscalib.calib(raw, constants, run=51)                    # in range -> silent
pscalib.calib(raw, constants, run=3)                     # stale -> StaleConstantsError
pscalib.calib(raw, constants, run=3, allow_stale=True)   # logs a warning, proceeds
```

With no `run=`, the check is skipped (the arithmetic is identical either way —
`allow_stale` only downgrades the refusal, it never changes the result).

### The uniform `Constants` contract

Whatever the provider, the apply path sees one small surface. The optional
`pscalib.Constants` adapter makes it explicit (wrapping is idempotent):

```python
C = pscalib.Constants.of(constants)
C.array("pedestals")     # the ndarray for a ctype
C.validities()           # {ctype: Validity} for staleness enforcement
C.det_type_hint          # the detector type the constants name themselves with
C.pin                    # the (detector_uniqueid, run) identity, if known
```

A plain `{ctype: ndarray}` dict, a psana-style `{ctype: (ndarray, meta)}` dict,
and a `CalibSnapshot` are all accepted directly — `Constants` is sugar, not a
requirement.

## Providers — where constants come from

All three feed the same `pscalib.calib(...)`; constants are byte-identical across
them, so the same raw frame yields the same calibrated frame (and image)
regardless of provider.

| Provider | Import | Deps | Network | Use |
|----------|--------|------|---------|-----|
| **Snapshot** (US-000) | `pscalib.load_snapshot(dir)` | numpy | none | reload constants captured once to disk |
| **Web-DB** (US-001) | `from pscalib.providers import webdb; webdb.get_constants(uniqueid, exp, run)` | `+requests,bson` (`web` extra) | on-site HTTP | fetch live from the calib web service |
| **BYO** | a `{ctype: ndarray}` / `{ctype: (ndarray, meta)}` dict | numpy | none | supply your own constants |

```python
# Web provider (the `web` extra; no psana, no DataSource, no Mongo socket):
from pscalib.providers import webdb
constants = webdb.get_constants(uniqueid, exp="mfx100848724", run=51)  # uniqueid = det.raw._uniqueid
calib = pscalib.calib(raw, constants)                                  # det type inferred
```

> **Off-site reachability (known gap).** The on-site endpoint
> (`https://psdmint.sdf.slac.stanford.edu/calib_ws/`) returns anonymously and is
> the correctness bar. The off-site endpoint
> (`https://psextapi.slac.stanford.edu/calib_ws/`) is **not routable from
> sdfiana025**; exercise and record the off-site read path from a genuinely
> off-site machine. Override the base URL with `LCLS_CALIB_HTTP`.

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
`{jungfrau,geometry,image,render}.py` calibration/assembly modules into
`pscalib` as the **canonical home**.

psdata's copies are **retained** (psdata is under active development — not
deleted), but to avoid two independently-editable copies, **psdata's calibration
modules are a re-export shim of `pscalib`** (chosen option (a)). There is
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

## Reference datasets

The acceptance tests regenerate psana ground truth themselves from these
(constants live in the tests, never in the library):

- **Jungfrau (auto-ranging, 3-gain):** `exp=mfx100848724 run=51
  dir=/sdf/data/lcls/ds/prj/public01/xtc det='jungfrau'` — raw `(32,512,1024)u16`;
  calib `(32,512,1024)f32`; image `(4216,4432)f32`;
  pedestals/pixel_gain/pixel_offset `(3,32,512,1024)f32`. Gain stage is in the raw
  bits (no Configure object needed).
- **epix10ka (7 gain ranges, config-driven):** `exp=ued1010667 run=177
  dir=/sdf/data/lcls/ds/prj/public01/xtc det='epixquad'` (class
  `epix10ka_raw_2_0_1`) — raw `(4,352,384)u16`; calib `(4,352,384)f32`;
  pedestals/pixel_gain `(7,4,352,384)f32`. The gain range is decoded from the
  per-ASIC `trbit`/`asicPixelConfig` Configure fields (psdata's
  `Run.seg_configs(detname)`, US-003) OR-ed with the per-event data gain bit —
  *not* from the calib DB. Pass that object as `config=` to `pscalib.calib`.

## Coverage gate

`tests/test_api_us005.py` is the multi-detector gate, all through the unified
public surface: jungfrau calib **and** assembled image, and epix10ka calib, each
byte-exact (`max|diff| == 0`) vs psana; plus a cross-provider check that the same
jungfrau image is produced whether the constants came from the snapshot provider
or the on-site web provider. The per-story tests (`test_calib_us000.py`,
`test_webdb_us001.py`, `test_validity_us002.py`, `test_epix10ka_us004.py`,
`test_no_drift_us000.py`) cover the layers beneath it.
