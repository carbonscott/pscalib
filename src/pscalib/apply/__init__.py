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

The derived constants (``poff``, ``gfac`` and the ``gfm`` fold) are cached
across calls, keyed on the identity of the arrays you pass -- that is where the
jungfrau speedup comes from, and it is unbounded and un-invalidatable by
construction (see :func:`pscalib.apply.jungfrau.calib_jungfrau` for the
contract).  The four functions that make it inspectable and reclaimable are
re-exported here so a caller never has to reach into a private module:

  * :func:`memo_clear` -- drop every entry (the escape hatch after an in-place
    mutation of a constants array, and the way to reclaim the footprint);
  * :func:`memo_nbytes` -- bytes of derived arrays held alive (403 MB - 1074 MB
    per jungfrau constants set; the table is in its docstring);
  * :func:`memo_size` -- number of live entries;
  * :func:`memo_stats` -- the counters (advisory: not atomic).
"""

from . import jungfrau  # noqa: F401
from . import epix10ka  # noqa: F401
from .jungfrau import calib_jungfrau
from .epix10ka import calib_epix10ka, mask_from_pixel_status
from ._fastcalib import (
    memo_clear, memo_nbytes, memo_size, memo_stats, last_call, backend_info,
)

__all__ = [
    "jungfrau", "calib_jungfrau",
    "epix10ka", "calib_epix10ka", "mask_from_pixel_status",
    "memo_clear", "memo_nbytes", "memo_size", "memo_stats",
    "last_call", "backend_info",
]
