#!/usr/bin/env python3
"""GATE-05 acceptance test: the RETRIEVAL oracle pscalib never had.

Why this gate exists
--------------------
``src/pscalib/providers/webdb.py`` is a 748-LOC, ``requests``-only re-derivation
of psana's web-DB read path -- it vendors, function-by-function, the closure of
``psana.pscalib.calib.MDBWebUtils.calib_constants_all_types`` (the exact call
psana itself makes to populate ``det.raw._calibconst``).  Its whole reason to
exist is the claim "byte-for-byte identical to what psana would have fetched".

Until this file, NOTHING in the suite tested that claim against psana's OWN
retrieval function.  The nearest gate (``test_webdb_us001.py``) compares
``webdb.get_constants`` to ``det.raw._calibconst`` -- the dict psana exposes
*after* opening a ``DataSource`` and constructing a ``Detector``.  That is a
fine end-to-end check, but the ORACLE it uses is one hop removed from the code
webdb.py actually re-derives: it goes through the detector-construction path, so
a drift between webdb.py and ``MDBWebUtils.calib_constants_all_types`` itself
could hide behind identical ``_calibconst`` output.  The "byte-exact retrieval"
PoC that *did* call ``MDBWebUtils`` directly called PSANA's copy, proving only
that the transport needs no framework -- it said nothing about the vendored copy
we ship.

So the RETRIEVAL half of pscalib (the 748 LOC in webdb.py) had no direct oracle.
This gate is that oracle: for the same ``(exp, run, detector-uniqueid)`` it calls

    pscalib :  webdb.get_constants(uniqueid, exp, run)
    psana   :  MDBWebUtils.calib_constants_all_types(uniqueid, exp=..., run=...)

-- the psana function webdb.py claims to reproduce -- and asserts, per ctype:

  * the two dicts return the SAME set of ctypes (a missing/extra ctype is itself
    a retrieval discrepancy -- a doc-selection or two-pass bug), and
  * each retrieved payload is BYTE-EQUAL (raw ``.tobytes()`` including NaN bit
    patterns for float ctypes; utf-8 bytes for str ctypes such as 'geometry'),
    cross-checked with a NaN-aware ``np.array_equal``.

Reference dataset (lives in the TEST, never in the library): jungfrau
``mfx100848724`` run 51, dir ``/sdf/data/lcls/ds/prj/public01/xtc``.

Environment
-----------
This is a psana-ORACLE gate: it needs the production psana env (psconda.sh) both
to read the detector ``_uniqueid`` and to call ``MDBWebUtils`` for the ground
truth, plus network access to the on-site calib web service (psdmint).  Run it
on the milano / sdfiana compute node via
``run_tests.sh tests/test_gate05_retrieval_oracle.py`` (it puts pscalib/src +
psdata/src on PYTHONPATH; ``import psana`` still resolves to the psconda env).

When psana is NOT importable the gate SKIPS through the HYG-05 marker protocol
(tests/_skips.py) under the slug ``gate05_retrieval_psana_oracle`` -- a slug that
is deliberately NOT in tests/skips_allowed.txt, so a run without psana FAILS
loudly (a skipped oracle is not a passing oracle) exactly like the other
``*_psana_*`` gates.  psana is the whole point: this suite exists to be compared
against it.
"""

import os
import sys

import numpy as np
import pytest

# --- machine-readable skip protocol (HYG-05); see tests/_skips.py -----------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from _skips import skip  # noqa: E402

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

# ctypes we insist BOTH sides return for this dataset (a sanity floor; the gate
# also compares every OTHER ctype either side returns, and asserts the full
# key-sets are identical).
CORE_CTYPES = ("pedestals", "pixel_gain", "pixel_offset",
               "pixel_status", "pixel_rms")


def _have_psana():
    try:
        import psana  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Byte-equality of a single retrieved payload.
#
# webdb.py and psana both build the payload with `np.frombuffer(gridfs_bytes,
# dtype).reshape(shape)` off the SAME blob, so a faithful re-derivation is
# byte-identical -- including NaN bit patterns, which `.tobytes()` preserves
# and which a naive `==` would drop.  We therefore compare (kind, shape, dtype,
# raw bytes); for float ndarrays we ALSO run a NaN-aware `np.array_equal` as a
# semantic cross-check (belt and suspenders: catches an equal-bytes-but-wrong-
# interpretation slip, and is the comparison the gate spec names explicitly).
# --------------------------------------------------------------------------
def _payload_signature(x):
    """(kind, shape, dtype-str, raw-bytes) -- a byte-exact fingerprint."""
    if isinstance(x, np.ndarray):
        return ("ndarray", tuple(x.shape), np.dtype(x.dtype).str,
                np.ascontiguousarray(x).tobytes())
    if isinstance(x, str):
        return ("str", (len(x),), "utf-8", x.encode("utf-8"))
    if isinstance(x, bytes):
        return ("bytes", (len(x),), "raw", x)
    # 'any'-typed / dict ctypes (e.g. xtcav): compare a stable repr as a last
    # resort -- not expected on the area-detector path this gate exercises.
    return (type(x).__name__, None, None, repr(x).encode("utf-8"))


def _array_equal_nan(a, b):
    """NaN-aware ndarray equality (the comparison the gate spec names)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:  # numpy < 1.19 has no equal_nan kwarg
            na, nb = np.isnan(a), np.isnan(b)
            return (bool(np.array_equal(na, nb)) and
                    bool(np.array_equal(a[~na], b[~nb])))
    return bool(np.array_equal(a, b))


def _payloads_byte_equal(g, o):
    """True iff pscalib payload ``g`` is byte-identical to psana payload ``o``."""
    if _payload_signature(g) != _payload_signature(o):
        return False
    if isinstance(g, np.ndarray) and isinstance(o, np.ndarray):
        # semantic cross-check on top of the byte fingerprint
        return _array_equal_nan(g, o)
    return True


# --------------------------------------------------------------------------
def _uniqueid_via_psana():
    """The detector long unique id -- ``det.raw._uniqueid`` -- the string psana
    passes as ``det`` to ``calib_constants_all_types``.  This (and reading the
    oracle) are the ONLY psana uses in this gate; the pscalib fetch opens no
    DataSource and imports no psana."""
    from psana import DataSource
    ds = DataSource(exp=EXP, run=RUN, dir=DIR)
    myrun = next(ds.runs())
    det = myrun.Detector(DET)
    return det.raw._uniqueid


@pytest.fixture
def uniqueid():
    """Under pytest, supply the same ``det.raw._uniqueid`` the ``__main__``
    runner computes for this module's EXP/RUN/DET (via psana)."""
    return _uniqueid_via_psana()


# --------------------------------------------------------------------------
# THE GATE: pscalib webdb vs psana MDBWebUtils, byte-for-byte, per ctype.
# --------------------------------------------------------------------------
def test_gate05_retrieval_oracle(uniqueid):
    """webdb.get_constants(...) == MDBWebUtils.calib_constants_all_types(...),
    byte-exact per ctype, for the reference jungfrau (exp, run)."""
    # ---- ORACLE: psana's OWN retrieval function (the code webdb.py vendors) --
    # Imported here (not at module top) so the module imports numpy-only when
    # psana is absent; the import path drags in mpi4py/dgram, which is exactly
    # why webdb.py vendored this instead of importing it.
    from psana.pscalib.calib import MDBWebUtils as wu
    oracle = wu.calib_constants_all_types(uniqueid, exp=EXP, run=RUN)
    assert oracle, ("psana MDBWebUtils.calib_constants_all_types returned "
                    "nothing for %s/%s %s -- calib web DB (psdmint) unreachable?"
                    % (EXP, RUN, uniqueid))
    for ct in CORE_CTYPES:
        assert ct in oracle, ("psana oracle did not return expected ctype %r "
                              "(got %s)" % (ct, sorted(oracle)))

    # ---- UNDER TEST: pscalib's vendored provider (NO DataSource, NO psana) --
    from pscalib.providers import webdb
    got = webdb.get_constants(uniqueid, exp=EXP, run=RUN)
    assert got, "pscalib webdb.get_constants returned nothing"

    # (1) KEY-SET IDENTITY.  A ctype present on one side but not the other is a
    # real retrieval discrepancy -- a doc-selection (select_doc_in_run_range) or
    # missing-types second-pass (calib_constants_of_missing_types) divergence.
    ok, gk = set(oracle), set(got)
    only_psana = sorted(ok - gk)
    only_pscalib = sorted(gk - ok)
    assert ok == gk, (
        "RETRIEVAL DISCREPANCY -- ctype key-sets differ.\n"
        "  present in psana ONLY  : %s\n"
        "  present in pscalib ONLY: %s" % (only_psana, only_pscalib))

    # (2) PER-CTYPE BYTE-EQUALITY of the retrieved payloads.
    mismatches = []
    for ct in sorted(ok):
        o_data = oracle[ct][0]
        g_data = got[ct][0]
        if _payloads_byte_equal(g_data, o_data):
            o_sig = _payload_signature(o_data)
            print("[gate05] %-16s %-8s %s  BYTE-EXACT == psana"
                  % (ct, o_sig[0], o_sig[1]))
        else:
            o_sig, g_sig = _payload_signature(o_data), _payload_signature(g_data)
            mismatches.append(ct)
            print("[gate05] %-16s MISMATCH  psana=%s/%s  pscalib=%s/%s"
                  % (ct, o_sig[0], o_sig[1], g_sig[0], g_sig[1]))

    assert not mismatches, (
        "RETRIEVAL DISCREPANCY -- pscalib webdb.get_constants is NOT byte-exact "
        "vs psana MDBWebUtils.calib_constants_all_types for ctype(s): %s"
        % mismatches)

    print("[gate05] pscalib webdb == psana MDBWebUtils for all %d ctypes "
          "(%s) -- the 748-LOC retrieval is byte-exact." % (len(ok), sorted(ok)))


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("GATE-05 acceptance: pscalib webdb retrieval vs psana MDBWebUtils "
          "oracle")
    print("=" * 72)

    if not _have_psana():
        skip("gate05_retrieval_psana_oracle",
             "psana not importable -- GATE-05 compares pscalib's webdb provider "
             "against psana's MDBWebUtils.calib_constants_all_types (the "
             "ORACLE) and reads the detector uniqueid via psana, so NOTHING in "
             "GATE-05 ran. Source psconda.sh on the milano/sdfiana node and run "
             "via run_tests.sh (this slug is intentionally not allow-listed, so "
             "a psana-less run fails loudly).")
        return

    uid = _uniqueid_via_psana()
    test_gate05_retrieval_oracle(uid)

    print("\nALL GATE-05 RETRIEVAL-ORACLE CHECKS PASSED")


if __name__ == "__main__":
    main()
