#!/usr/bin/env python3
"""US-002 acceptance test: validity-range / staleness ENFORCEMENT (refuse-by-default).

Verifies the US-002 acceptance criteria:

  (1) MODEL -- ``pscalib.model.Validity`` parses each constant's run-validity
      range from its metadata doc (``run`` = first valid run, ``run_end`` = last,
      sentinel ``'end'`` => open-ended), matching psana's selection rule
      ``select_doc_in_run_range`` (begin <= rnum <= end; 'end' -> rnum_max 9999).
      Snapshots and web fetches carry the pin (detector_uniqueid, run) plus the
      per-ctype validity.

  (2) ENFORCEMENT (the one correctness feature beyond psdata's advisory
      ``is_valid_for_run``) -- applying constants pinned at run R to raw from a
      run R' OUTSIDE [run, run_end]:
        * RAISES ``StaleConstantsError`` BY DEFAULT,
        * ``allow_stale=True`` downgrades the refusal to a LOGGED WARNING,
        * an IN-RANGE apply passes SILENTLY.
      All three outcomes are demonstrated observably -- first fully offline
      (no psana), then on the real jungfrau dataset through ``HDRImager``.

  (3) REGRESSION -- with enforcement satisfied (in-range), the jungfrau apply
      still produces calib (32,512,1024) f32 with max|diff| == 0 vs
      ``det.raw.calib(evt)`` for mfx100848724/r51 (US-000's number is unchanged).

  (4) IMPORT PURITY -- ('psana','mpi4py','h5py','dgram','pymongo') absent from
      sys.modules after the apply (fresh interpreter), with enforcement wired in.

The offline checks (1, 2-offline, 4) run WITHOUT psana.  The psana cross-checks
(2-live, 3) need the production psana env (psconda.sh) and skip cleanly with a
message otherwise.  Run on sdfiana025 via ``run_tests.sh tests/test_validity_us002.py``.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# --- locate the pscalib package (parent of this tests dir) ------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reference dataset -- lives in the TEST, never in the library.
EXP = "mfx100848724"
RUN = 51
DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
DET = "jungfrau"

FORBIDDEN = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


# ==========================================================================
# (1) MODEL: Validity parses run/run_end + 'end' sentinel, matches psana
# ==========================================================================
def test_validity_model():
    from pscalib.model import (Validity, Pin, StaleConstantsError,
                               validities_from_calibconst)

    # closed range
    v = Validity(10, 20)
    assert (v.run, v.run_end, v.open_ended) == (10, 20, False), v
    assert not v.contains(9) and v.contains(10) and v.contains(20)
    assert not v.contains(21)

    # 'end' sentinel -> open-ended, capped at RUN_MAX (psana CalibDoc.rnum_max)
    vo = Validity(5, "end")
    assert vo.open_ended and vo.run_end == Validity.RUN_MAX == 9999, vo
    assert vo.contains(5) and vo.contains(9999) and not vo.contains(4)
    # None also means open-ended
    assert Validity(5, None).open_ended

    # digit string run_end accepted (psana stores it as int or str)
    assert Validity(3, "7").run_end == 7

    # from_meta reads the doc fields psana's CalibDoc reads
    assert Validity.from_meta({"run": 51, "run_end": "end"}) == Validity(51, "end")
    assert Validity.from_meta({"run": 51, "run_end": 99}) == Validity(51, 99)
    # missing run is not a parseable validity
    try:
        Validity.from_meta({"run_end": 99})
        raise AssertionError("expected KeyError for missing 'run'")
    except KeyError:
        pass

    # as_dict round-trips the sentinel
    assert Validity(5, "end").as_dict() == {"run": 5, "run_end": "end"}
    assert Validity(5, 9).as_dict() == {"run": 5, "run_end": 9}

    # Pin carries the (uniqueid, run) identity
    p = Pin("jungfrau_serial_xyz", 51, detname="jungfrau", exp="mfx100848724")
    assert p.detector_uniqueid == "jungfrau_serial_xyz" and p.run == 51
    assert Pin.from_snapshot_pin(
        {"detector_uniqueid": "u", "run": 7, "detname": "d", "exp": "e"}) \
        == Pin("u", 7)

    # validities_from_calibconst maps a {ctype:(data,meta)} dict to {ctype:Validity}
    cc = {
        "pedestals": (np.zeros(1), {"run": 40, "run_end": "end"}),
        "pixel_gain": (np.zeros(1), {"run": 50, "run_end": 60}),
        "no_meta": (np.zeros(1), {}),                       # skipped (no run)
    }
    vs = validities_from_calibconst(cc)
    assert set(vs) == {"pedestals", "pixel_gain"}, vs
    assert vs["pedestals"] == Validity(40, "end")
    assert vs["pixel_gain"] == Validity(50, 60)
    print("[ok] (1) Validity model: run/run_end/'end' sentinel + from_meta + "
          "Pin + validities_from_calibconst")


def test_validity_matches_psana_calibdoc():
    """The Validity.contains test must agree with psana's run-range rule
    (CalibDoc / select_doc_in_run_range: begin <= rnum <= end, 'end'->9999).
    Verified against pscalib's VENDORED CalibDoc (the byte-faithful copy of
    psana's), so this runs without the psana env."""
    from pscalib.model import Validity
    from pscalib.providers.webdb import CalibDoc

    cases = [
        ({"run": 10, "run_end": 20, "_id": "0" * 24}, [5, 10, 15, 20, 25]),
        ({"run": 5, "run_end": "end", "_id": "0" * 24}, [4, 5, 100, 9999]),
    ]
    for doc, runs in cases:
        cd = CalibDoc(doc)
        assert cd.valid, doc
        v = Validity.from_meta(doc)
        for r in runs:
            psana_in = (cd.begin <= r <= cd.end)
            assert v.contains(r) == psana_in, (
                f"Validity.contains({r}) != CalibDoc range for {doc}: "
                f"pscalib={v.contains(r)} psana={psana_in}")
    print("[ok] (1) Validity.contains agrees with psana's vendored CalibDoc "
          "run-range rule (begin<=rnum<=end, 'end'->9999)")


# ==========================================================================
# (2) ENFORCEMENT, fully offline: check_validity all three outcomes
# ==========================================================================
def test_check_validity_three_outcomes():
    from pscalib.model import Validity, Pin, StaleConstantsError, check_validity

    pin = Pin("u", 100, detname="jungfrau")
    # constants valid for runs [50, 200]
    validities = {
        "pedestals": Validity(50, 200),
        "pixel_gain": Validity(50, 200),
    }

    # (a) IN RANGE -> silent (returns empty offender list, no raise, no warn)
    out = check_validity(validities, 100, allow_stale=False, pin=pin)
    assert out == [], out
    print("[ok] (2a) in-range check_validity(run=100) passed SILENTLY")

    # (b) OUT OF RANGE, default -> raises StaleConstantsError
    try:
        check_validity(validities, 9000, allow_stale=False, pin=pin)
        raise AssertionError("expected StaleConstantsError for run 9000")
    except StaleConstantsError as e:
        assert e.run == 9000
        offending = {ct for ct, _ in e.offenders}
        assert offending == {"pedestals", "pixel_gain"}, e.offenders
        assert "STALE" in str(e) and "allow_stale=True" in str(e)
        print(f"[ok] (2b) out-of-range check_validity(run=9000) RAISED "
              f"StaleConstantsError: {e}")

    # (c) OUT OF RANGE, allow_stale=True -> logged WARNING, returns offenders
    logger = logging.getLogger("pscalib.model")
    records = []

    class _Capture(logging.Handler):
        def emit(self, rec):
            records.append(rec)

    h = _Capture()
    logger.addHandler(h)
    prev = logger.level
    logger.setLevel(logging.WARNING)
    try:
        out = check_validity(validities, 9000, allow_stale=True, pin=pin,
                             log=logger)
    finally:
        logger.removeHandler(h)
        logger.setLevel(prev)
    assert len(out) == 2, out
    assert any(r.levelno == logging.WARNING and "STALE" in r.getMessage()
               for r in records), [r.getMessage() for r in records]
    print(f"[ok] (2c) out-of-range allow_stale=True LOGGED A WARNING (no raise): "
          f"{records[-1].getMessage()}")


def _write_synthetic_snapshot(snap_dir, ped_validity=("end",), run=100):
    """Write a minimal but real pscalib snapshot dir: a (3,1,4,4) pedestals +
    pixel_gain + an (1,4,4) mask, with the given per-ctype validity, so
    CalibSnapshot.check_validity / HDRImager can exercise enforcement offline."""
    os.makedirs(snap_dir, exist_ok=True)
    ped = np.zeros((3, 1, 4, 4), dtype=np.float32)
    gain = np.ones((3, 1, 4, 4), dtype=np.float32)
    mask = np.ones((1, 4, 4), dtype=np.uint8)
    np.save(os.path.join(snap_dir, "pedestals.npy"), ped, allow_pickle=False)
    np.save(os.path.join(snap_dir, "pixel_gain.npy"), gain, allow_pickle=False)
    np.save(os.path.join(snap_dir, "mask.npy"), mask, allow_pickle=False)
    run_end = ped_validity[0]
    manifest = {
        "schema": "psdata.calib.snapshot/v1",
        "pin": {"detname": "jungfrau", "detector_uniqueid": "synthetic_uid",
                "run": run, "exp": "synthexp", "dir": "/none"},
        "files": {"pedestals": "pedestals.npy", "pixel_gain": "pixel_gain.npy",
                  "mask": "mask.npy"},
        "geometry_file": None,
        "validity": {
            "pedestals": {"run": 50, "run_end": run_end, "version": "v1"},
            "pixel_gain": {"run": 50, "run_end": run_end, "version": "v1"},
        },
        "shapes": {},
    }
    with open(os.path.join(snap_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


def test_snapshot_enforcement_offline():
    """CalibSnapshot.check_validity enforces the same three outcomes on a real
    (synthetic) snapshot directory, fully offline."""
    from pscalib.providers.snapshot import load_snapshot
    from pscalib.model import StaleConstantsError, Validity, Pin

    tmp = tempfile.mkdtemp(prefix="pscalib_us002_snap_")
    try:
        # pedestals/pixel_gain valid for runs [50, 200] (run_end=200)
        snap_dir = os.path.join(tmp, "jungfrau_r0100")
        _write_synthetic_snapshot(snap_dir, ped_validity=(200,), run=100)
        snap = load_snapshot(snap_dir)

        # snapshot carries the pin + per-ctype Validity (acceptance: "carry the
        # pin plus the per-ctype validity")
        assert isinstance(snap.pin_obj, Pin)
        assert snap.pin_obj.detector_uniqueid == "synthetic_uid"
        assert snap.pin_obj.run == 100
        vs = snap.validities()
        assert vs["pedestals"] == Validity(50, 200), vs
        assert vs["pixel_gain"] == Validity(50, 200), vs
        # validity_obj typed accessor
        assert snap.validity_obj("pedestals") == Validity(50, 200)

        # in range -> silent
        assert snap.check_validity(150) == []
        # out of range -> raises by default
        try:
            snap.check_validity(500)
            raise AssertionError("expected StaleConstantsError for run 500")
        except StaleConstantsError as e:
            assert e.run == 500 and e.pin.detector_uniqueid == "synthetic_uid"
        # allow_stale -> warns, returns offenders
        offenders = snap.check_validity(500, allow_stale=True)
        assert {ct for ct, _ in offenders} == {"pedestals", "pixel_gain"}
        print("[ok] (2) CalibSnapshot offline enforcement: in-range silent, "
              "out-of-range raises by default, allow_stale warns")

        # is_valid_for_run (advisory) still works and agrees with in/out range
        assert snap.is_valid_for_run(150) is True
        assert snap.is_valid_for_run(500) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# (3) REGRESSION + live enforcement on the real jungfrau dataset (psana)
# ==========================================================================
def test_jungfrau_inrange_regression_and_enforcement(out_dir):
    """With enforcement satisfied (in-range), the jungfrau apply still produces
    calib (32,512,1024) f32 with max|diff| == 0 vs psana (US-000's number); and
    an out-of-range run raises by default / warns with allow_stale=True."""
    import pscalib
    import pscalib.providers.snapshot as ps_snap
    import pscalib.geometry as pgeo
    from pscalib.model import StaleConstantsError, Validity

    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    evt = next(myrun.events())
    gt_raw = np.asarray(det.raw.raw(evt))
    gt_calib = np.asarray(det.raw.calib(evt))
    assert gt_raw.shape == (32, 512, 1024) and gt_raw.dtype == np.uint16
    assert gt_calib.shape == (32, 512, 1024) and gt_calib.dtype == np.float32

    snap_dir = ps_snap.snapshot_calib(exp=EXP, run=RUN, dir=DIR, detname=DET,
                                      out_dir=out_dir)
    pgeo.cache_pixel_indexes_for_snapshot(snap_dir)
    snap = ps_snap.load_snapshot(snap_dir)
    imager = pscalib.HDRImager(snap, derive_geometry_if_missing=False)

    # the real constants' validity for this run -- the snapshot's pedestals must
    # cover RUN (psana selected them for it), so in-range enforcement must pass.
    vs = snap.validities()
    assert "pedestals" in vs, vs
    for ct, v in vs.items():
        assert v.contains(RUN), (
            f"{ct} validity {v} does not cover the snapshot run {RUN} -- "
            f"psana would not have returned it")
    print(f"[validity] per-ctype ranges all cover run {RUN}: "
          f"{ {ct: str(v) for ct, v in vs.items()} }")

    # IN-RANGE enforced apply -> byte-exact (US-000's number, with run= passed)
    my_calib = imager.calib(gt_raw, run=RUN)          # enforced, in range
    assert my_calib.shape == (32, 512, 1024) and my_calib.dtype == np.float32
    dcal = np.abs(np.nan_to_num(my_calib) - np.nan_to_num(gt_calib))
    assert np.array_equal(my_calib, gt_calib), (
        f"in-range enforced calib not byte-exact: max|diff|={dcal.max()}")
    print(f"[byte-exact] in-range enforced calib(run={RUN}) {my_calib.shape} "
          f"max|diff|={dcal.max()} array_equal=True (regression OK)")

    # the zero-arg call (no enforcement) still equals it (backward compat)
    assert np.array_equal(imager.calib(gt_raw), gt_calib)

    # OUT-OF-RANGE -> raises by default.  Find a run strictly beyond every
    # constant's run_end (or below every run) so it's guaranteed stale.
    max_end = max(v.run_end for v in vs.values())
    stale_run = max_end + 1 if max_end < Validity.RUN_MAX else 0
    if stale_run > Validity.RUN_MAX:
        stale_run = 0  # below the first valid run instead
    # ensure it's actually out of range for at least one ctype
    assert any(not v.contains(stale_run) for v in vs.values()), (
        f"could not construct an out-of-range run (ranges: {vs})")
    raised = False
    try:
        imager.calib(gt_raw, run=stale_run)
    except StaleConstantsError as e:
        raised = True
        print(f"[ok] (2) live out-of-range calib(run={stale_run}) RAISED "
              f"StaleConstantsError (refuse-by-default)")
    assert raised, f"expected StaleConstantsError for stale run {stale_run}"

    # allow_stale=True -> warns, still computes byte-exact (constants unchanged)
    stale_calib = imager.calib(gt_raw, run=stale_run, allow_stale=True)
    assert np.array_equal(stale_calib, gt_calib), (
        "allow_stale apply changed the arithmetic -- it must only downgrade "
        "the refusal, not the math")
    print(f"[ok] (2) live out-of-range calib(run={stale_run}, allow_stale=True) "
          f"WARNED and still computed (byte-exact, math unchanged)")
    return snap_dir


# ==========================================================================
# (4) IMPORT PURITY: enforcement wired in, fresh interp, forbidden set absent
# ==========================================================================
def test_offline_import_purity_with_enforcement():
    """In a FRESH interpreter: import pscalib, build a synthetic snapshot,
    exercise the staleness enforcement (raise + allow_stale), and apply the
    jungfrau decode -- none of the five forbidden modules may appear."""
    tmp = tempfile.mkdtemp(prefix="pscalib_us002_purity_")
    snap_dir = os.path.join(tmp, "jungfrau_r0100")
    try:
        _write_synthetic_snapshot(snap_dir, ped_validity=(200,), run=100)
        code = (
            "import sys, numpy as np\n"
            "import pscalib\n"
            "from pscalib import load_snapshot, calib_jungfrau\n"
            "from pscalib.model import StaleConstantsError\n"
            f"snap = load_snapshot({snap_dir!r})\n"
            # enforcement: in-range silent, out-of-range raises, allow_stale warns
            "assert snap.check_validity(150) == []\n"
            "raised = False\n"
            "try:\n"
            "    snap.check_validity(500)\n"
            "except StaleConstantsError:\n"
            "    raised = True\n"
            "assert raised, 'expected StaleConstantsError'\n"
            "assert snap.check_validity(500, allow_stale=True)\n"
            # apply the jungfrau decode on synthetic raw (1 segment, 4x4 toy)
            "raw = np.zeros((1, 4, 4), dtype=np.uint16)\n"
            "calib = calib_jungfrau(raw, snap.pedestals, snap.pixel_gain, "
            "None, snap.mask)\n"
            "assert calib.shape == (1, 4, 4) and calib.dtype == np.float32\n"
            "pscalib.assert_no_framework_imports()\n"
            f"bad = [m for m in {FORBIDDEN!r} if m in sys.modules]\n"
            "assert not bad, bad\n"
            "assert 'numpy' in sys.modules\n"
            "print('CLEAN')\n"
        )
        env = dict(os.environ)
        inherited = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (_PKG_PARENT + (os.pathsep + inherited
                                            if inherited else ""))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env)
        assert out.returncode == 0, (
            "purity subprocess failed:\nSTDOUT:%s\nSTDERR:%s"
            % (out.stdout, out.stderr))
        assert "CLEAN" in out.stdout, out.stdout
        print("[ok] (4) fresh interp: enforcement + jungfrau apply clean of "
              + str(FORBIDDEN))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("US-002 acceptance: validity-range / staleness ENFORCEMENT")
    print("=" * 72)

    # (1) model + (2 offline) enforcement + (4) purity -- no psana needed
    test_validity_model()
    test_validity_matches_psana_calibdoc()
    test_check_validity_three_outcomes()
    test_snapshot_enforcement_offline()
    test_offline_import_purity_with_enforcement()

    if not _have_psana():
        print("\n[skip] psana not importable -- live regression + live "
              "enforcement gate skipped.  Source psconda.sh on sdfiana025.")
        print("\nUS-002 offline checks PASSED (psana-dependent checks skipped)")
        return

    tmp = tempfile.mkdtemp(prefix="pscalib_us002_")
    try:
        test_jungfrau_inrange_regression_and_enforcement(
            out_dir=os.path.join(tmp, "regr"))
        print("[ok] (3) in-range jungfrau apply byte-exact vs psana "
              "(max|diff| == 0) + live enforcement (raise/warn)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL US-002 ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
