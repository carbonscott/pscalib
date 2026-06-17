"""pscalib.apply -- pure-numpy per-detector calibration gain decode.

Each detector type is a leaf "apply" function that turns a raw stack plus its
calibration constants into a calibrated ADU stack, byte-exact vs
``det.raw.calib(evt)``.  No psana, no DB, no MPI -- numpy only.

  * :mod:`pscalib.apply.jungfrau`  -- Jungfrau 3-gain decode (gain in the
    raw bits).  Established in US-000 (lifted from psdata, already byte-exact).
  * :mod:`pscalib.apply.epix10ka`  -- NEW in US-004: per-pixel 7-gain-range
    decode driven by the per-ASIC Configure object.

The agreed thin abstraction (US-005) is a plugin ``plugin(raw, constants,
config=None) -> calib`` registered by detector type in
:mod:`pscalib.registry`; jungfrau and epix10ka are two leaf plugins.
"""

from . import jungfrau  # noqa: F401
from . import epix10ka  # noqa: F401
from .jungfrau import calib_jungfrau
from .epix10ka import calib_epix10ka, mask_from_pixel_status

__all__ = [
    "jungfrau", "calib_jungfrau",
    "epix10ka", "calib_epix10ka", "mask_from_pixel_status",
]
