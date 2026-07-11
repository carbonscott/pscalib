"""Fresh per-run scratch directories for ``__main__``-mode tests (HYG-05).

A test suite that only passes on a clean machine is not a suite.

``tests/test_geometry_us006.py`` used to hard-code its ``__main__`` scratch
directory::

    out_dir = os.environ.get("PSCALIB_TEST_OUT", "/tmp/pscalib_us006_out")
    os.makedirs(out_dir, exist_ok=True)

and then handed it to ``pscalib.providers.snapshot.snapshot_calib(...)``, whose
default is ``overwrite=False``::

    if os.path.isdir(snap_dir) and os.listdir(snap_dir) and not overwrite:
        raise FileExistsError(...)

So the FIRST run populated ``/tmp/pscalib_us006_out/jungfrau_r0051/`` and every
run after that died with ``FileExistsError`` -- unless a human remembered to
``rm -rf /tmp/pscalib_us006_out`` first.  The suite was not idempotent.

The fix is at the source: hand each run a directory that is *guaranteed* fresh
and empty, so ``snapshot_calib`` can never collide with its own leftovers.  No
manual cleanup, no ``overwrite=True`` (which would have weakened the very
collision guard ``snapshot_calib`` exists to provide), no ``rm -rf`` in the
runner (which would have to guess at paths it does not own).

The sibling tests (US-000/002/004/005/008) already did exactly this with
``tempfile.mkdtemp``; this module just makes the pattern shared, honours the
``PSCALIB_TEST_OUT`` knob, and is import-cheap (stdlib only) so the HYG-05
hygiene regression test can exercise it without numpy, pytest or psana.
"""

import os
import shutil
import tempfile

#: If set, scratch dirs are created *underneath* this directory instead of the
#: system temp dir.  Note the change of meaning: it is now the PARENT of the
#: per-run dir, not the per-run dir itself.  That is what makes it safe to
#: point at a persistent location -- runs can never tread on each other.
BASE_ENV = "PSCALIB_TEST_OUT"

#: If set (to anything non-empty), :func:`cleanup_out_dir` keeps the directory
#: instead of removing it, so a failed run's artifacts can be inspected.
KEEP_ENV = "PSCALIB_KEEP_TEST_OUT"


def make_out_dir(prefix):
    """Create and return a FRESH, EMPTY scratch directory for this run.

    Every call returns a distinct, newly created, empty directory -- calling it
    twice in a row (i.e. running the suite twice in a row) can never collide.

    Parameters
    ----------
    prefix : str
        Short tag for the directory name, e.g. ``"pscalib_us006"``.

    Returns
    -------
    str
        Absolute path of a directory that exists and is empty.
    """
    base = os.environ.get(BASE_ENV) or None
    if base:
        os.makedirs(base, exist_ok=True)
        return tempfile.mkdtemp(prefix=prefix + "_", dir=base)
    return tempfile.mkdtemp(prefix=prefix + "_")


def cleanup_out_dir(path):
    """Remove a directory made by :func:`make_out_dir` (unless ``KEEP_ENV``).

    Safe to call from a ``finally:`` block; never raises.
    """
    if not path:
        return
    if os.environ.get(KEEP_ENV):
        print("[out_dir] kept for inspection (%s set): %s" % (KEEP_ENV, path))
        return
    shutil.rmtree(path, ignore_errors=True)
