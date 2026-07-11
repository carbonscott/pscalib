"""Machine-readable SKIP protocol for the pscalib acceptance suite (HYG-05).

A skip is not a pass.

The cross-check tests in this suite are dual-mode: under ``pytest`` they are
ordinary test functions, and under ``python3 tests/test_x.py`` a ``main()``
runner calls them in order.  Historically, when an oracle prerequisite was
absent (psana not importable, psdata not on ``PYTHONPATH``, ...) a test would
print a human-readable ``[skip] ...`` line, ``return``, and **exit 0**.  The
runner only looked at exit statuses, so a suite whose entire psana oracle had
silently evaporated still reported "green".  That is precisely the
``psconda PYTHONPATH clobber`` failure mode: put ``<repo>/src`` *ahead* of the
psana env and every byte-exact gate skips itself into a false pass.

The fix is to make every skip **visible to the runner** by printing a marker
line that ``run_tests.sh`` can count:

    ##SKIP## <name> :: <reason>

``run_tests.sh`` tallies these, prints them in the final summary, and checks
each ``<name>`` against ``tests/skips_allowed.txt``.  Any skip that is not
explicitly justified in that allowlist FAILS the run.  Tests keep their old
control flow -- they still ``return`` (or call ``pytest.skip``) -- they just
can no longer do it invisibly.

Usage::

    from _skips import skip

    if not _have_psana():
        skip("us004_psana_byte_exact_gate",
             "psana not importable -- byte-exact gate needs the psconda env")
        return
"""

import sys

#: The token ``run_tests.sh`` greps for.  Keep it in sync with the runner.
SKIP_MARKER = "##SKIP##"


def skip(name, reason):
    """Announce a skipped check on stdout in the machine-readable form.

    Parameters
    ----------
    name : str
        A stable, unique slug for this skip site (e.g.
        ``"us006_render_via_vendored_derivation_no_psana"``).  This is the key
        ``tests/skips_allowed.txt`` is matched on, so it must not drift.
    reason : str
        Human-readable explanation of why the check could not run.

    Returns
    -------
    None
        Returned so call sites may write ``return skip(...)`` if they like.

    Notes
    -----
    This only *reports*; it does not alter control flow.  The caller stays
    responsible for returning (or raising ``pytest.skip``) exactly as before.
    """
    # Single line, flushed, so it survives the runner's `tee` and interleaves
    # sanely with any stderr the test also produces.
    print("%s %s :: %s" % (SKIP_MARKER, name, reason))
    sys.stdout.flush()
    return None
