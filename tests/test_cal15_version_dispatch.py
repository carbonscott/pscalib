#!/usr/bin/env python3
"""CAL-15 regression test: version-aware calib dispatch (refuse, don't guess).

THE BUG (CAL-15).  pscalib dispatches calibration on the detector FAMILY token
(``epix10ka_raw_3_0_1`` -> ``epix10ka``) and has exactly ONE apply plugin per
family.  psana, however, chooses the calib ALGORITHM from the full VERSION
TRIPLE -- the deployed release ships two epix10ka calib implementations
(``calib_epix10ka_v02`` for the newer class, the legacy ``calib_epix10ka_any``
for classes that don't override).  Raw decode is self-describing; calibration is
NOT.  Before the fix, ``registry.detector_type_of`` *threw the version token
away* at the point of dispatch, so ``pscalib.calib('epix10ka_raw_9_9_9', ...)``
silently applied the one family plugin to a version it was never byte-exactness
validated against -- returning a calibrated array with NO signal that it was
guessing.

THE FIX (what this test proves).  pscalib keeps the family normalization for
LOOKUP but records, per family, the version triples its single plugin has
actually been validated against, and REFUSES (raises) when handed an
unvalidated version rather than guessing.  Same class of fix as psdata's DET-10:
"know what you don't know."

THIS TEST is the pre-fix/post-fix discriminator.  It contains NO part of the
fix.  It is self-contained -- numpy only, NO psana, NO SLAC data -- so it runs
anywhere and is robust to cwd.  It drives the real ``pscalib.calib(...)`` apply
entry point (which routes through ``get_plugin`` / ``detector_type_of``) with
valid-shaped SYNTHETIC constants, and asserts:

  (A) a KNOWN-VALIDATED version (``epix10ka_raw_2_0_1`` / ``jungfrau_raw_0_1_0``
      -- the classes the byte-exact oracle gates use) still dispatches and
      returns a proper calibrated array (guards against over-refusal breaking
      the byte-exact oracle, AND proves the synthetic constants are well-formed
      so the only variable below is the version triple); and

  (B) a plausibly-real but UNVALIDATED version (``epix10ka_raw_9_9_9`` /
      ``jungfrau_raw_9_9_9``) is REFUSED, not silently calibrated.

On the PARENT commit (family dispatch, version discarded) case (B) returns an
array with no signal -> the "must refuse" assertion fails -> exit 1.  On the
FIX it refuses -> exit 0.
"""

import os
import sys

import numpy as np

# --- locate the pscalib package (parent of this tests dir), cwd-robust -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(os.path.dirname(_HERE), "src")  # .../pscalib/src
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import pscalib            # noqa: E402
import pscalib.registry as reg   # noqa: E402

# Reference shapes (epix10ka: 7 gain ranges over 4 segments; jungfrau: 3 ranges
# over 32 segments) -- exactly what the oracle tests use, so a KNOWN-validated
# version really does dispatch and the ONLY variable is the version triple.
EPIX_RAW_SHAPE = (4, 352, 384)
EPIX_CONS_SHAPE = (7, 4, 352, 384)
JF_RAW_SHAPE = (32, 512, 1024)
JF_CONS_SHAPE = (3, 32, 512, 1024)

# The version triples the byte-exact oracle gates actually measured against.
EPIX_VALIDATED = "epix10ka_raw_2_0_1"    # test_epix10ka_us004 / _trbit_us008
JF_VALIDATED = "jungfrau_raw_0_1_0"      # test_calib_us000 / test_purity_us007
# Plausibly-real but NOT in the validated set (a hypothetical newer deploy).
EPIX_UNVALIDATED = "epix10ka_raw_9_9_9"
JF_UNVALIDATED = "jungfrau_raw_9_9_9"


class _Ns:
    """Stand-in for a per-ASIC Configure namespace (trbit / asicPixelConfig)."""
    def __init__(self, trbit, apc):
        self.trbit = trbit
        self.asicPixelConfig = apc


class _Seg:
    """Stand-in for psdata's per-segment Configure object (has ``.config``)."""
    def __init__(self, config):
        self.config = config


def _epix_config():
    """A valid, non-empty synthetic epix10ka per-segment config (4 segments)."""
    return {i: _Seg(_Ns(np.zeros(4, np.uint8),
                        np.zeros((4, 176, 192), np.uint8))) for i in range(4)}


def _epix_constants():
    return {"pedestals": np.zeros(EPIX_CONS_SHAPE, np.float32),
            "pixel_gain": np.ones(EPIX_CONS_SHAPE, np.float32)}


def _jf_constants():
    return {"pedestals": np.zeros(JF_CONS_SHAPE, np.float32),
            "pixel_gain": np.ones(JF_CONS_SHAPE, np.float32)}


def _dispatch(det_class, raw, constants, config=None):
    """Drive ``pscalib.calib(det_class, raw, constants, ...)`` and report the
    OUTCOME as a behavior tuple -- no dependence on the fix's exception type.

    Returns ``("returned", array)`` if calib dispatched and produced a result
    (the silent-guess behavior), or ``("refused", exc)`` if it raised.
    """
    try:
        out = pscalib.calib(det_class, raw, constants, config=config)
    except Exception as e:            # noqa: BLE001 -- any refusal counts here
        return ("refused", e)
    return ("returned", out)


# --------------------------------------------------------------------------
# (A) a KNOWN-validated version must still dispatch (no over-refusal) --------
#     also proves the synthetic constants are well-formed, so the only
#     variable in (B) is the version triple.
# --------------------------------------------------------------------------
def test_validated_version_still_dispatches():
    outcome, obj = _dispatch(EPIX_VALIDATED, np.zeros(EPIX_RAW_SHAPE, np.uint16),
                             _epix_constants(), config=_epix_config())
    assert outcome == "returned", (
        f"the KNOWN-validated version {EPIX_VALIDATED!r} must dispatch (no false "
        f"refusal -- that would break the byte-exact oracle); got {outcome!r} "
        f"{obj!r}. If this fails, the synthetic constants/config are malformed, "
        f"which is a test bug, not a CAL-15 signal.")
    assert getattr(obj, "shape", None) == EPIX_RAW_SHAPE, obj
    assert obj.dtype == np.float32, obj.dtype

    outcome, obj = _dispatch(JF_VALIDATED, np.zeros(JF_RAW_SHAPE, np.uint16),
                             _jf_constants())
    assert outcome == "returned", (
        f"the KNOWN-validated version {JF_VALIDATED!r} must dispatch; got "
        f"{outcome!r} {obj!r}")
    assert getattr(obj, "shape", None) == JF_RAW_SHAPE, obj
    print(f"[ok] (A) validated versions dispatch: {EPIX_VALIDATED} -> "
          f"{EPIX_RAW_SHAPE} f32, {JF_VALIDATED} -> {JF_RAW_SHAPE}")


# --------------------------------------------------------------------------
# (B) an UNVALIDATED version must be REFUSED, not silently calibrated --------
#     THIS is the pre-fix/post-fix discriminator.
# --------------------------------------------------------------------------
def test_unvalidated_version_is_refused():
    outcome, obj = _dispatch(EPIX_UNVALIDATED,
                             np.zeros(EPIX_RAW_SHAPE, np.uint16),
                             _epix_constants(), config=_epix_config())
    assert outcome == "refused", (
        f"CAL-15 REGRESSION: pscalib SILENTLY applied its single epix10ka "
        f"plugin to the UNVALIDATED version {EPIX_UNVALIDATED!r} and returned a "
        f"calibrated array {getattr(obj, 'shape', obj)!r} instead of refusing. "
        f"Family dispatch discarded the version triple; psana would pick the "
        f"calib algorithm FROM that triple (the release ships >1 epix10ka calib "
        f"implementation), so applying the one plugin here is a silent guess. "
        f"pscalib must refuse (or signal) when handed a version it never "
        f"validated its plugin against.")

    # Below here only runs post-fix (pre-fix already failed above).  Verify the
    # refusal is the dedicated, actionable CAL-15 signal -- not some unrelated
    # error that would let a broken fix pass.
    exc = obj
    UErr = getattr(reg, "UnvalidatedCalibVersionError", None)
    if UErr is not None:
        assert isinstance(exc, UErr), (
            f"expected an UnvalidatedCalibVersionError, got "
            f"{type(exc).__name__}: {exc}")
    msg = str(exc)
    assert EPIX_UNVALIDATED in msg, (
        f"refusal must NAME the offending class {EPIX_UNVALIDATED!r}; got: {msg}")
    assert "epix10ka" in msg, (
        f"refusal must name the family 'epix10ka'; got: {msg}")
    assert "plugin_epix10ka" in msg or "unvalidated" in msg.lower(), (
        f"refusal must name the plugin it WOULD have used / flag it unvalidated; "
        f"got: {msg}")

    # a second family, for robustness: jungfrau's unvalidated version too
    outcome_jf, obj_jf = _dispatch(JF_UNVALIDATED,
                                   np.zeros(JF_RAW_SHAPE, np.uint16),
                                   _jf_constants())
    assert outcome_jf == "refused", (
        f"CAL-15 REGRESSION: pscalib silently calibrated the UNVALIDATED "
        f"jungfrau version {JF_UNVALIDATED!r} (returned "
        f"{getattr(obj_jf, 'shape', obj_jf)!r}) instead of refusing.")
    if UErr is not None:
        assert isinstance(obj_jf, UErr), obj_jf

    print(f"[ok] (B) unvalidated versions refused: {EPIX_UNVALIDATED!r} and "
          f"{JF_UNVALIDATED!r} raised {type(exc).__name__} naming the class, "
          f"family, and plugin")


# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CAL-15 regression: version-aware calib dispatch (refuse, don't guess)")
    print("=" * 72)
    test_validated_version_still_dispatches()
    test_unvalidated_version_is_refused()
    print("\nCAL-15 REGRESSION TEST PASSED "
          "(validated versions dispatch; unvalidated versions refused)")


if __name__ == "__main__":
    main()
