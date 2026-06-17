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
    NOTE (US-007): the compiled xtc2 reader is the SUBMODULE ``psana.dgram``,
    so the guard must prefix-match -- ``"psana"`` catches every ``psana.*`` and
    ``"mpi4py"`` catches ``mpi4py.MPI`` -- not merely the bare leaf names (an
    exact-name check could never see ``psana.dgram``).  See
    :func:`assert_no_framework_imports`.
  * ``h5py`` -- the HDF5 transport (dead LCLS1 / XTCAV path); not on the
    area-detector apply/geometry path.
  * ``pymongo`` -- the Mongo client.  Calib *reads* go over HTTP (``requests``)
    and never open a Mongo socket; ``pymongo`` appearing would mean we took the
    framework's DB path by mistake.

The web retrieval path (US-001) may import ``requests`` and ``bson`` (the
pure-python ``ObjectId``); those are NOT forbidden.  ``bson`` is a different
package from ``pymongo`` -- importing ``bson`` does not open a Mongo connection.

psana is permitted only inside ONE LAZY function-body import, a one-time prep
step run in the psana env, never on the offline/apply path:

  * :func:`pscalib.providers.snapshot.snapshot_calib` -- ``from psana import
    DataSource`` (capture the constants once).

The geometry pixel-index derivation
(:func:`pscalib.geometry.pixel_coord_indexes_from_text`) USED to be a second
lazy psana touch; as of US-006 it uses the vendored numpy-only
:mod:`pscalib._geometry` closure and imports no psana at all, so the whole
apply/render path is now framework-free.
"""

import sys

#: Modules that must never appear in ``sys.modules`` after importing pscalib and
#: running an offline path (reload / apply) or a web fetch in a fresh
#: interpreter.  EXTENDS psdata's ``('psana','mpi4py','h5py')`` with
#: ``dgram`` + ``pymongo``.
FORBIDDEN_MODULES = ("psana", "mpi4py", "h5py", "dgram", "pymongo")


def _leaked_modules(forbidden):
    """Return the sorted ``sys.modules`` keys that violate ``forbidden``.

    A loaded module *leaks* if its dotted name EQUALS a forbidden root OR is a
    SUBMODULE of one (``name == root`` or ``name.startswith(root + ".")``).  The
    submodule rule is the whole point of US-007: the real compiled xtc2 reader
    is the SUBMODULE ``psana.dgram`` (and ``mpi4py.MPI`` for MPI), which appears
    in ``sys.modules`` under its dotted key, never as a bare leaf -- so an
    exact-name check could never catch it.  Prefix-matching ``root + "."`` makes
    ``"psana"`` catch every ``psana.*`` and ``"mpi4py"`` catch ``mpi4py.MPI``.
    """
    out = []
    for name in sys.modules:
        for root in forbidden:
            if name == root or name.startswith(root + "."):
                out.append(name)
                break
    return sorted(out)


def assert_no_framework_imports(forbidden=FORBIDDEN_MODULES):
    """Raise ``AssertionError`` if any forbidden module (or a submodule of one)
    leaked into ``sys.modules``.

    The snapshot capture imports psana lazily *on call*; merely importing
    pscalib, reloading a snapshot, applying constants, deriving geometry index
    maps (now vendored, US-006), or doing a web fetch must not.  Call this after
    one of those offline/web operations to assert the path stayed clean.

    Matching is by dotted-name prefix (see :func:`_leaked_modules`): a forbidden
    root catches itself AND every submodule under it, so ``"psana"`` flags
    ``psana.dgram`` / ``psana.psexp`` / any ``psana.*`` and ``"mpi4py"`` flags
    ``mpi4py.MPI`` -- not just the bare top-level package.
    """
    leaked = _leaked_modules(forbidden)
    assert not leaked, (
        f"pscalib offline/web path must not import {tuple(forbidden)} "
        f"(or any submodule); found {leaked} in sys.modules")
