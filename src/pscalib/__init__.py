"""pscalib -- standalone, pure-Python LCLS-II calibration package.

The calibration sibling to the :mod:`psdata` reader, exactly as ``psdata`` is
the framework-free sibling to psana's xtc2 reader.  pscalib *retrieves*
calibration constants (a snapshot-to-disk provider, or -- from US-001 -- a
``requests``-only web-DB client; never the psana framework / MPI / BYOA) and
*applies* them in pure numpy, byte-exact vs psana.

Importing :mod:`pscalib` pulls in only numpy.  No psana, no mpi4py, no dgram,
no h5py, no pymongo (see :mod:`pscalib._purity`).  The web retrieval provider
needs ``requests`` + ``bson`` (the ``web`` extra) and is imported explicitly
(``import pscalib.providers.webdb``) when you want it, never at package import.

Layout (US-000 establishes the offline engine; later stories fill the rest)::

    pscalib/
      apply/jungfrau.py       # Jungfrau 3-gain HDR gain decode (== det.raw.calib)
      apply/epix10ka.py       # NEW (US-004)
      providers/snapshot.py   # capture (lazy psana) + numpy reload of constants
      providers/webdb.py      # NEW (US-001): requests-only web-DB client
      geometry.py             # geometry text -> per-pixel image index maps
      image.py                # pixel-array -> 2-D image remap (== det.raw.image)
      render.py               # HDRImager: raw -> calib -> image, fully offline
      model.py                # NEW (US-001/US-002): Constants / Pin / Validity
      registry.py             # NEW (US-004/US-005): det_type -> apply plugin

The offline calibration engine (US-000), lifted from psdata's already
byte-exact ``calib`` + ``hdr`` layers::

    from pscalib.providers.snapshot import load_snapshot   # pure numpy
    from pscalib.render import HDRImager                   # pure numpy

    snap   = load_snapshot("snapshots/jungfrau_r0051")     # reload constants
    imager = HDRImager(snap)                               # offline render engine
    calib, image = imager.render(raw_stack)               # numpy only

This package retains its own canonical copy of the calibration engine; psdata's
``calib``/``hdr`` modules re-export from here (one canonical implementation, no
drift).  See the README "psdata relationship" section.
"""

from . import _purity  # noqa: F401
from ._purity import FORBIDDEN_MODULES, assert_no_framework_imports

# The constants contract + staleness enforcement (US-002).  Pure-python; carries
# only validity metadata (run-ranges + pin), never arrays -- so importing it adds
# no dependency beyond the stdlib.
from . import model     # noqa: F401
from .model import (
    StaleConstantsError,
    Validity,
    Pin,
    Constants,
    check_validity,
    validities_from_calibconst,
    detector_type_hint,
)

# The pure-numpy apply / geometry / image / render engine (US-000).  These pull
# in only numpy at import time; the psana touches (snapshot capture, geometry
# derivation) are lazy function-body imports inside the relevant functions.
from . import apply       # noqa: F401  (apply.jungfrau, ...)
from . import geometry    # noqa: F401
from . import image       # noqa: F401
from . import render      # noqa: F401
from . import providers   # noqa: F401  (providers.snapshot; webdb is opt-in)
from . import registry    # noqa: F401  (det_type -> apply plugin dispatch; US-004)

from .apply.jungfrau import calib_jungfrau
from .apply.epix10ka import calib_epix10ka, mask_from_pixel_status
from .geometry import (
    pixel_coord_indexes_from_text,
    cache_pixel_indexes_for_snapshot,
    load_pixel_indexes,
)
from .image import assemble_image
from .render import HDRImager, from_snapshot_dir
from .providers.snapshot import (
    CalibSnapshot,
    load_snapshot,
    snapshot_calib,
)
from .registry import (
    calib,
    register,
    get_plugin,
    registered_types,
    detector_type_of,
    detector_type_for_constants,
)

__all__ = [
    "apply", "geometry", "image", "render", "providers", "model", "registry",
    "FORBIDDEN_MODULES", "assert_no_framework_imports",
    "StaleConstantsError", "Validity", "Pin", "Constants", "check_validity",
    "validities_from_calibconst", "detector_type_hint",
    "calib_jungfrau",
    "calib_epix10ka", "mask_from_pixel_status",
    "registry", "calib", "register", "get_plugin", "registered_types",
    "detector_type_of", "detector_type_for_constants",
    "pixel_coord_indexes_from_text", "cache_pixel_indexes_for_snapshot",
    "load_pixel_indexes",
    "assemble_image",
    "HDRImager", "from_snapshot_dir",
    "CalibSnapshot", "load_snapshot", "snapshot_calib",
]
