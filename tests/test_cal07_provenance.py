#!/usr/bin/env python3
"""CAL-07 regression: the snapshot must persist the source-document identity.

Bug (CAL-07): the calib web DB is *mutable* -- the same
``(exp, run, detector, ctype)`` query can later resolve to a DIFFERENT document
if someone re-deploys constants.  psana attaches, per constant, a stable
document identity: ``_id`` (the immutable Mongo ObjectId of that exact document)
and ``time_stamp`` (when it was deployed).  The snapshot's metadata filter
(``pscalib.providers.snapshot._slim_meta``, driven by the ``_META_KEEP``
whitelist) used to DROP ``_id`` as "DB-internal bookkeeping".  Without ``_id``,
a snapshotted/published number can no longer be traced back to the exact
document that produced it -- so it is not reproducible, defeating the point of a
snapshot.

The fix retains ``_id`` (as a stable JSON string; an ObjectId is not natively
JSON-serializable) and ``time_stamp`` in the filtered metadata, while still
dropping genuinely non-load-bearing bookkeeping (``cwd`` / ``host`` / ...).

This test is fully self-contained: stdlib + numpy only, NO psana, NO SLAC data,
NO network.  It targets the pure metadata-retention function ``_slim_meta``
directly (``snapshot_calib`` itself needs psana to *capture*, but the retention
logic under test does not), plus the on-disk round-trip through
``load_snapshot`` / ``CalibSnapshot.validity``.

Discriminator: on the PARENT (pre-fix) commit ``_id`` is dropped from the
filtered metadata, so the ``_id``-retention assertions raise and the process
exits non-zero.  On the fix they pass.  ``time_stamp`` was already retained on
both, so ``_id`` is the field that flips.

Run: ``python3 tests/test_cal07_provenance.py`` (from anywhere -- cwd-robust).
"""

import json
import os
import sys
import tempfile

import numpy as np  # noqa: F401  (dependency parity with the numpy-only suite)

# --- locate the pscalib package (parent of this tests dir), cwd-robust ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pscalib.providers.snapshot import (  # noqa: E402
    MANIFEST_NAME,
    _slim_meta,
    load_snapshot,
)

#: A realistic 24-char hex Mongo ObjectId string (what str(ObjectId) yields).
_OID_HEX = "5aa9c2f9c9b5f0a1b2c3d4e5"
_TIME_STAMP = "20180314_215937"


class _FakeObjectId:
    """Stand-in for ``bson.objectid.ObjectId`` (avoids importing bson).

    ``str()`` returns the 24-char hex id -- the stable identity psana's own
    ``sec_and_ts_from_id`` treats as the document's id -- but the object itself
    is NOT natively JSON-serializable: ``json.dumps`` raises ``TypeError`` on it
    unless the filter coerces it to a string first.  This mirrors exactly why
    ``_id`` cannot simply be whitelisted verbatim.
    """

    def __init__(self, hexid):
        self._hex = hexid

    def __str__(self):
        return self._hex

    def __repr__(self):
        return "_FakeObjectId(%r)" % self._hex


def _synthetic_meta(oid):
    """A psana-style per-ctype metadata doc: the real validity/provenance fields
    plus DB-internal bookkeeping that must be dropped."""
    return {
        # validity range + constant identity (retained today)
        "run": 5,
        "run_end": "end",
        "version": 3,
        "ctype": "pedestals",
        "detector": "jungfrau_000003",
        "detname": "jungfrau",
        "dettype": "jungfrau",
        "experiment": "mfx100848724",
        "time_sec": 1521064777,
        # DOCUMENT IDENTITY -- the CAL-07 fields that make a number reproducible
        "_id": oid,
        "time_stamp": _TIME_STAMP,
        # genuinely non-load-bearing DB bookkeeping -- must be dropped
        "cwd": "/sdf/home/someone/calibman",
        "host": "psanaphi105",
        "uid": 31054,
    }


# --------------------------------------------------------------------------
def test_id_and_time_stamp_retained_bookkeeping_dropped():
    """``_slim_meta`` keeps the document identity (``_id`` + ``time_stamp``) and
    drops ``cwd`` / ``host`` -- and the kept ``_id`` is JSON-serializable."""
    oid = _FakeObjectId(_OID_HEX)

    # Guard the premise: a raw ObjectId stand-in really is NOT JSON-serializable.
    try:
        json.dumps({"_id": oid})
    except TypeError:
        pass
    else:  # pragma: no cover -- would mean the stand-in is unrealistic
        raise AssertionError("stand-in ObjectId must not be JSON-serializable")

    slim = _slim_meta(_synthetic_meta(oid))

    # (a) DOCUMENT IDENTITY retained -- this is what flips pre/post fix.
    assert "_id" in slim, (
        "CAL-07: _slim_meta dropped '_id' -- the snapshot cannot identify the "
        "source document, so a published number is not reproducible. "
        "kept keys=%s" % sorted(slim))
    assert "time_stamp" in slim, "CAL-07: '_id' present but 'time_stamp' dropped"

    # (b) _id survives as a STABLE STRING (the 24-char hex), not a raw ObjectId.
    assert isinstance(slim["_id"], str), (
        "_id must be persisted as a stable string, got %r" % type(slim["_id"]))
    assert slim["_id"] == _OID_HEX, (
        "_id string must equal str(ObjectId); got %r" % (slim["_id"],))
    assert slim["time_stamp"] == _TIME_STAMP

    # (c) genuinely useless bookkeeping still dropped.
    for junk in ("cwd", "host", "uid"):
        assert junk not in slim, "%r should be dropped, not retained" % junk

    # (d) the retained validity/provenance fields are still there.
    for keep in ("run", "run_end", "version", "ctype", "detname"):
        assert keep in slim, "regression: dropped a validity field %r" % keep

    # (e) the WHOLE filtered dict is directly JSON-serializable (a raw ObjectId
    #     in it would raise TypeError here).
    encoded = json.dumps(slim)
    assert _OID_HEX in encoded, "the _id hex must appear in the JSON"

    print("[ok] _slim_meta retains _id(str)+time_stamp, drops cwd/host, "
          "JSON-serializable")


def test_plain_string_id_preserved():
    """If psana already handed us a plain-string ``_id`` (some paths do), it is
    preserved verbatim -- coercion is idempotent, not lossy."""
    meta = _synthetic_meta(_OID_HEX)  # _id is already a str here
    slim = _slim_meta(meta)
    assert slim.get("_id") == _OID_HEX
    json.dumps(slim)  # still serializable
    print("[ok] plain-string _id preserved verbatim")


def test_missing_id_is_backward_compatible():
    """An OLD-style doc with no ``_id`` (pre-provenance snapshots) filters
    without error and simply carries no ``_id`` -- unknown provenance, not a
    crash.  Keeps reading legacy snapshots backward-compatible."""
    meta = _synthetic_meta(_FakeObjectId(_OID_HEX))
    del meta["_id"]
    slim = _slim_meta(meta)          # must not raise
    assert "_id" not in slim
    assert "time_stamp" in slim      # other provenance still kept
    json.dumps(slim)
    print("[ok] missing _id -> filtered cleanly (backward-compatible read)")


def test_on_disk_round_trip_via_load_snapshot():
    """End-to-end: what ``_slim_meta`` produces, written into a manifest and
    reloaded with ``load_snapshot``, comes back with ``_id`` + ``time_stamp``
    intact -- so the snapshot on disk can identify its source document.

    Uses ONLY public reload surface (``load_snapshot`` / ``validity``) that
    exists on both the parent and the fix, so this discriminates purely on
    whether ``_slim_meta`` kept ``_id``.
    """
    slim = _slim_meta(_synthetic_meta(_FakeObjectId(_OID_HEX)))
    manifest = {
        "schema": "psdata.calib.snapshot/v1",
        "pin": {
            "detname": "jungfrau",
            "detector_uniqueid": "jungfrau_000003",
            "run": 51,
            "exp": "mfx100848724",
            "dir": "/sdf/data/lcls/ds/prj/public01/xtc",
        },
        "files": {},              # no arrays needed to exercise the round-trip
        "geometry_file": None,
        "validity": {"pedestals": slim},
        "shapes": {},
    }
    tmp = tempfile.mkdtemp(prefix="pscalib_cal07_")
    try:
        with open(os.path.join(tmp, MANIFEST_NAME), "w") as fh:
            json.dump(manifest, fh)          # serializable only if _id is a str
        snap = load_snapshot(tmp)
        v = snap.validity("pedestals")
        assert v.get("_id") == _OID_HEX, (
            "CAL-07: reloaded snapshot lost '_id' -- the on-disk artifact "
            "cannot be traced to its source document. validity=%s" % v)
        assert v.get("time_stamp") == _TIME_STAMP
        for junk in ("cwd", "host"):
            assert junk not in v
        print("[ok] on-disk round-trip preserves _id + time_stamp")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CAL-07 regression: snapshot persists source-document identity "
          "(_id/time_stamp)")
    print("=" * 72)
    test_id_and_time_stamp_retained_bookkeeping_dropped()
    test_plain_string_id_preserved()
    test_missing_id_is_backward_compatible()
    test_on_disk_round_trip_via_load_snapshot()
    print("\nALL CAL-07 CHECKS PASSED")


if __name__ == "__main__":
    main()
