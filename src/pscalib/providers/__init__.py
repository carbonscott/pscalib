"""pscalib.providers -- pluggable calibration-constant *providers*.

A provider answers "give me the constants for (detector, run)".  The apply
engine (:mod:`pscalib.apply`) is provider-agnostic; constants can come from any
of these and are applied identically:

  * :mod:`pscalib.providers.snapshot` -- snapshot-to-disk.  Capture a
    detector's ``_calibconst`` once (the only psana-using step, a lazy import),
    then reload + apply fully offline with numpy only.  Established in US-000.
  * :mod:`pscalib.providers.webdb` -- NEW in US-001: a ``requests``-only client
    over the calib web service (no psana, byte-exact).  Imported explicitly
    (it needs the ``web`` extra: ``requests`` + ``bson``); NOT imported here so
    that importing :mod:`pscalib.providers` stays numpy-only.
  * :mod:`pscalib.providers.byo` -- NEW (US-005): the caller supplies a
    constants dict / snapshot dir directly (numpy only).

Only the snapshot provider is re-exported at package import; ``webdb`` is
opt-in to keep the import surface numpy-only.
"""

from . import snapshot  # noqa: F401
from .snapshot import (
    CalibSnapshot,
    load_snapshot,
    snapshot_calib,
)

__all__ = ["snapshot", "CalibSnapshot", "load_snapshot", "snapshot_calib"]
