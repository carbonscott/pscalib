"""pscalib._purity -- the shared import-purity rule.

The whole point of pscalib is that the *offline* paths (reload a snapshot,
apply constants, assemble an image) and the *web* retrieval path stay free of
the psana framework, MPI, and the Mongo/HDF5 transports.  This module is the
single source of truth for that rule.

The forbidden set EXTENDS psdata's original ``_FORBIDDEN_MODULES``
(``('psana','mpi4py','h5py')``) with ``dgram`` and ``pymongo``:

  * ``psana`` / ``mpi4py`` / ``dgram`` -- the framework + its C-extension xtc2
    reader + MPI.  Any ``psana.*`` import runs ``psana/__init__.py`` which pulls
    all three, so the web-DB retrieval functions are VENDORED, not imported.
  * ``h5py`` -- the HDF5 transport (dead LCLS1 / XTCAV path); not on the
    area-detector apply/geometry path.
  * ``pymongo`` -- the Mongo client.  Calib *reads* go over HTTP (``requests``)
    and never open a Mongo socket; ``pymongo`` appearing would mean we took the
    framework's DB path by mistake.

The web retrieval path (US-001) may import ``requests`` and ``bson`` (the
pure-python ``ObjectId``); those are NOT forbidden.  ``bson`` is a different
package from ``pymongo`` -- importing ``bson`` does not open a Mongo connection.

psana is permitted only inside two LAZY function-body imports, both one-time
prep steps run in the psana env, never on the offline/apply path:

  * :func:`pscalib.providers.snapshot.snapshot_calib` -- ``from psana import
    DataSource`` (capture the constants once).
  * :func:`pscalib.geometry.pixel_coord_indexes_from_text` --
    ``from psana.pscalib.geometry.GeometryAccess import GeometryAccess``
    (derive the pixel index maps once).
"""

import sys

#: Modules that must never appear in ``sys.modules`` after importing pscalib and
#: running an offline path (reload / apply) or a web fetch in a fresh
#: interpreter.  EXTENDS psdata's ``('psana','mpi4py','h5py')`` with
#: ``dgram`` + ``pymongo``.
FORBIDDEN_MODULES = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def assert_no_framework_imports(forbidden=FORBIDDEN_MODULES):
    """Raise ``AssertionError`` if any forbidden module leaked into
    ``sys.modules``.

    The snapshot capture and the one-time geometry derivation import psana
    lazily *on call*; merely importing pscalib, reloading a snapshot, applying
    constants, or doing a web fetch must not.  Call this after one of those
    offline/web operations to assert the path stayed clean.
    """
    leaked = [m for m in forbidden if m in sys.modules]
    assert not leaked, (
        f"pscalib offline/web path must not import {tuple(forbidden)}; "
        f"found {leaked} in sys.modules")
