#!/usr/bin/env python3
"""HYG-05 regression: run_tests.sh must be a METER, not a rubber stamp.

The bug this pins down had three heads, all in the runner:

  1. NO TALLY.  The runner looped, kept the worst exit status, and exited. No
     pass count, no fail count, and -- fatally -- no SKIP count. The tests in
     this suite skip by printing a message and RETURNING, so they exit 0. A run
     in which psana was not importable and every byte-exact oracle gate skipped
     itself still exited 0 and was recorded as green. (That is exactly the
     `psconda PYTHONPATH clobber` trap: replace PYTHONPATH instead of prepending
     to it, psana disappears, every gate skips, suite is "green", nothing was
     ever compared.)

  2. AN INCOMPLETE DEFAULT LIST.  tests/test_no_drift_us000.py existed on disk
     but was not in the runner's hand-maintained TESTS array, so a default
     full-suite invocation never ran it. (The same file was once silently
     DELETED by a stray edit and the deletion rode into an unrelated commit --
     which a runner that only reads its own array cannot notice either.)

  3. NO IDEMPOTENCY.  test_geometry_us006.py wrote to a FIXED scratch path and
     snapshot_calib() refuses to overwrite a non-empty snapshot dir, so every
     full-suite run needed a manual `rm -rf /tmp/pscalib_us006_out` first.

This test is deliberately SELF-CONTAINED: stdlib only. No psana, no SLAC data,
no network, no numpy, no pytest. It drives run_tests.sh with synthetic probe
scripts in a temp dir, so it runs anywhere, in milliseconds, and pins the
runner's *logic* rather than the science.

Dual-mode, like its siblings:
    python3 tests/test_runner_hygiene_hyg05.py     # or via run_tests.sh
    pytest tests/test_runner_hygiene_hyg05.py
"""

import ast
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS_DIR)
_RUNNER = os.path.join(_REPO, "run_tests.sh")
_ALLOW_FILE = os.path.join(_TESTS_DIR, "skips_allowed.txt")

if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from _outdir import make_out_dir  # noqa: E402
from _skips import SKIP_MARKER  # noqa: E402

#: Matches the runner's tally line, e.g. "3 passed, 1 failed, 2 skipped".
_SUMMARY_RE = re.compile(r"(\d+) passed, (\d+) failed, (\d+) skipped")

#: The scratch dir test_geometry_us006.py used to reuse across runs, which is
#: why every full-suite run needed a manual `rm -rf` of it first.
_OLD_FIXED_DIR = "/tmp/pscalib_us006_out"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mask(text):
    """Neutralise skip markers/idioms before we ever print captured output.

    This test file is itself part of the default suite, so it runs *inside*
    run_tests.sh. If we echoed a probe's raw output (or let an assertion message
    carry it into a traceback), the OUTER runner would grep our stdout, find a
    marker (or a bare `[skip]`/`SKIPPED` line) we merely quoted, and count a
    phantom skip. Everything we print or embed in an assertion goes through here
    first, so a hygiene FAILURE never pollutes the outer runner's skip scan.
    """
    text = text.replace(SKIP_MARKER, "<SKIP-MARKER>")
    return re.sub(r"\[skip\]|skipping|skipped", "<skip-idiom>", text,
                  flags=re.IGNORECASE)


def _write_probe(dirpath, name, body):
    """Write a synthetic probe test script and return its path."""
    path = os.path.join(dirpath, name)
    with open(path, "w") as fh:
        fh.write(body)
    return path


def _skip_probe(slug, reason="synthetic probe: pretending an oracle is absent"):
    """A probe that emits ONE skip marker and then exits 0 -- the exact shape of
    the false pass this whole ticket is about."""
    return (
        "import sys\n"
        "print(%r + ' ' + %r + ' :: ' + %r)\n"
        "sys.exit(0)\n" % (SKIP_MARKER, slug, reason)
    )


_PASS_PROBE = "import sys\nprint('probe: nothing to report')\nsys.exit(0)\n"
_FAIL_PROBE = "import sys\nprint('probe: deliberate failure')\nsys.exit(3)\n"

# The pre-fix bug pattern in its purest form: a probe that ANNOUNCES a skip the
# OLD way and exits 0, emitting NO ##SKIP## marker. Built from concatenated
# fragments so this source file itself carries no literal "[skip]" token that a
# scanner (ours or a reviewer's) could mistake for a real one.
_BARE_SKIP_PROBE = (
    "import sys\n"
    "print('[' + 'skip' + '] no psana -- old idiom, exits 0, no marker')\n"
    "sys.exit(0)\n"
)


def _run_runner(args, cwd=None):
    """Invoke run_tests.sh from an unrelated cwd; return (rc, combined output).

    Output is CAPTURED (never inherited), so probe skip markers cannot leak into
    our own stdout and be miscounted by an outer runner.
    """
    proc = subprocess.run(
        ["bash", _RUNNER] + list(args),
        cwd=cwd or tempfile.gettempdir(),   # robust to cwd: never run from $REPO
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _allowlisted_names():
    """The skip names tests/skips_allowed.txt forgives."""
    names = []
    if not os.path.isfile(_ALLOW_FILE):
        return names
    with open(_ALLOW_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or " :: " not in line:
                continue
            names.append(line.split(" :: ", 1)[0])
    return names


# ---------------------------------------------------------------------------
# (1) THE DISCRIMINATOR: an unlisted skip must FAIL the run
# ---------------------------------------------------------------------------
def test_unlisted_skip_fails_the_run():
    """A test that emits an unlisted ##SKIP## marker and exits 0 must turn the
    suite RED.

    On the parent commit the runner only looked at exit statuses, so this probe
    -- which exits 0 -- made it exit 0. That is the false pass. Here it must
    exit nonzero and name the offending skip.
    """
    tmp = tempfile.mkdtemp(prefix="hyg05_unlisted_")
    try:
        slug = "hyg05_probe_unlisted_skip"
        probe = _write_probe(tmp, "probe_unlisted_skip.py", _skip_probe(slug))
        rc, out = _run_runner([probe])

        assert rc != 0, (
            "runner exited 0 despite an UNJUSTIFIED skip -- a skip is not a "
            "pass (HYG-03/HYG-05). Output:\n" + _mask(out))
        expected = ("UNJUSTIFIED SKIP: %s -- a skip is not a pass "
                    "(HYG-03/HYG-05)" % slug)
        assert expected in out, (
            "runner did not report the unjustified skip verbatim; expected\n  "
            + expected + "\nin:\n" + _mask(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[hyg05] an unlisted skip turns the suite RED (exit != 0)")


def test_bare_unrouted_skip_line_fails_the_run():
    """A probe that prints an OLD-idiom skip line and exits 0 -- with NO
    ##SKIP## marker -- must turn the suite RED.

    This is the THIRD discriminator. The marker protocol only meters skips that
    route through skip(); a bare `print("[skip] ...") ; return` bypasses it
    entirely. On the parent commit (and on a fix that only counted markers) such
    a probe exits 0 and scores a silent PASS -- the very bug pattern this ticket
    exists to kill. The runner must scan the captured logs for the idiom and
    fail.
    """
    tmp = tempfile.mkdtemp(prefix="hyg05_bare_")
    try:
        probe = _write_probe(tmp, "probe_bare_skip.py", _BARE_SKIP_PROBE)
        rc, out = _run_runner([probe])

        assert rc != 0, (
            "runner exited 0 on a bare unrouted skip line (old idiom, no "
            "marker) -- the skip protocol can be bypassed by simply not using "
            "it (HYG-05). Output:\n" + _mask(out))
        assert "UNJUSTIFIED SKIP: <unrouted>" in out, (
            "runner did not flag the unrouted skip as unjustified. Output:\n"
            + _mask(out))
        # And it must point the author at the fix.
        assert "skip(name, reason)" in out, (
            "runner did not tell the author to route the skip through "
            "skip(name, reason). Output:\n" + _mask(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[hyg05] a printed skip-idiom that bypasses skip() turns the suite RED")


def test_summary_reports_an_explicit_skip_count():
    """The summary must tally passed/failed/SKIPPED explicitly, every run."""
    tmp = tempfile.mkdtemp(prefix="hyg05_summary_")
    try:
        probes = [
            _write_probe(tmp, "probe_pass.py", _PASS_PROBE),
            _write_probe(tmp, "probe_skip.py",
                         _skip_probe("hyg05_probe_counted_skip")),
        ]
        rc, out = _run_runner(probes)

        m = _SUMMARY_RE.search(out)
        assert m is not None, (
            "runner printed no 'N passed, M failed, S skipped' tally -- it is "
            "not a meter. Output:\n" + _mask(out))
        passed, failed, skipped = (int(g) for g in m.groups())
        assert (passed, failed, skipped) == (2, 0, 1), (
            "wrong tally: got %d passed, %d failed, %d skipped; expected "
            "2 passed, 0 failed, 1 skipped (both probes exit 0; one skips). "
            "Output:\n%s" % (passed, failed, skipped, _mask(out)))
        # Both probes exit 0, so the ONLY thing that can redden this run is the
        # skip -- and it must.
        assert rc != 0, (
            "two exit-0 probes, one of them skipping, still exited 0 -- the "
            "skip was scored as a pass. Output:\n" + _mask(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # NOTE: this line runs INSIDE the full suite, so its own text must not read
    # as an unrouted skip to the runner's bare-skip scan -- keep it free of the
    # literal idiom words "[skip]"/"skipping"/"skipped".
    print("[hyg05] summary carries an explicit passed / failed / skip tally")


def test_failing_test_is_counted_and_reddens_the_run():
    """The meter must count failures too (and not stop at the first one)."""
    tmp = tempfile.mkdtemp(prefix="hyg05_fail_")
    try:
        probes = [
            _write_probe(tmp, "probe_fail.py", _FAIL_PROBE),
            _write_probe(tmp, "probe_pass.py", _PASS_PROBE),
        ]
        rc, out = _run_runner(probes)

        m = _SUMMARY_RE.search(out)
        assert m is not None, "no tally printed. Output:\n" + _mask(out)
        passed, failed, skipped = (int(g) for g in m.groups())
        assert (passed, failed, skipped) == (1, 1, 0), (
            "wrong tally: got %d passed, %d failed, %d skipped; expected "
            "1 passed, 1 failed, 0 skipped. The pass AFTER the failure must "
            "still have run. Output:\n%s"
            % (passed, failed, skipped, _mask(out)))
        assert rc != 0, "a failing test did not redden the run"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[hyg05] failures are counted, and one red test does not abort the rest")


def test_allowlisted_skip_is_forgiven():
    """A skip named in tests/skips_allowed.txt is reported, but does not fail.

    This is the other half of the contract: the allowlist must actually forgive,
    otherwise 'justified skip' would be a fiction and people would delete the
    check instead of justifying it.
    """
    allowed = _allowlisted_names()
    if not allowed:
        # An empty allowlist is a legitimate, deliberate state (see
        # skips_allowed.txt). There is then nothing to forgive, and
        # test_unlisted_skip_fails_the_run already pins the important half.
        print("[hyg05] allowlist is empty; nothing to forgive (that is fine)")
        return

    tmp = tempfile.mkdtemp(prefix="hyg05_allowed_")
    try:
        probe = _write_probe(tmp, "probe_allowed_skip.py",
                             _skip_probe(allowed[0]))
        rc, out = _run_runner([probe])

        m = _SUMMARY_RE.search(out)
        assert m is not None, "no tally printed. Output:\n" + _mask(out)
        passed, failed, skipped = (int(g) for g in m.groups())
        assert (passed, failed, skipped) == (1, 0, 1), (
            "wrong tally for an allowlisted skip: %d/%d/%d. Output:\n%s"
            % (passed, failed, skipped, _mask(out)))
        assert rc == 0, (
            "an ALLOWLISTED skip (%r) still failed the run -- the allowlist "
            "does not forgive. Output:\n%s" % (allowed[0], _mask(out)))
        assert "UNJUSTIFIED SKIP" not in out, (
            "allowlisted skip was reported as unjustified. Output:\n"
            + _mask(out))
        assert "justification:" in out, (
            "runner did not print the justification for the skip. Output:\n"
            + _mask(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[hyg05] an allowlisted skip is reported with its justification and "
          "does NOT fail the run")


# ---------------------------------------------------------------------------
# (2) THE SECOND DISCRIMINATOR: the default list must cover the tests on disk
# ---------------------------------------------------------------------------
def _tests_on_disk():
    return sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(_TESTS_DIR, "test_*.py")))


def test_default_list_covers_every_test_on_disk():
    """Every tests/test_*.py must be in the runner's default suite.

    Fails on the parent commit: tests/test_no_drift_us000.py is on disk but is
    absent from run_tests.sh's TESTS array, so a bare `bash run_tests.sh` never
    ran it.
    """
    on_disk = _tests_on_disk()
    assert on_disk, "no tests/test_*.py found -- did the glob break?"

    with open(_RUNNER) as fh:
        runner_src = fh.read()

    missing = [t for t in on_disk if t not in runner_src]
    assert not missing, (
        "run_tests.sh's default suite does not mention %d test file(s) that "
        "exist on disk: %s -- they would never run in a default full-suite "
        "invocation (HYG-05)." % (len(missing), ", ".join(missing)))
    print("[hyg05] run_tests.sh names all %d tests/test_*.py on disk"
          % len(on_disk))


def test_runner_reports_its_default_suite_and_it_matches_disk():
    """`run_tests.sh --list` must resolve to exactly the tests on disk.

    The stronger, structural form of the check above: the runner itself is asked
    what it would run, and the answer must equal the filesystem. This is what
    makes the omission class of bug impossible rather than merely fixed once.
    """
    rc, out = _run_runner(["--list"])
    assert rc == 0, ("`run_tests.sh --list` failed (rc=%d) -- the runner cannot "
                     "report its own default suite. Output:\n%s"
                     % (rc, _mask(out)))
    listed = sorted(ln.strip() for ln in out.splitlines()
                    if ln.strip().startswith("test_") and ln.strip().endswith(".py"))
    on_disk = _tests_on_disk()
    assert listed == on_disk, (
        "the runner's default suite and tests/test_*.py on disk disagree.\n"
        "  runner : %s\n  on disk: %s" % (listed, on_disk))
    print("[hyg05] `run_tests.sh --list` == tests/test_*.py on disk (%d files)"
          % len(on_disk))


# ---------------------------------------------------------------------------
# (3) IDEMPOTENCY: no manual `rm -rf` between runs, ever
# ---------------------------------------------------------------------------
def test_scratch_dirs_are_fresh_every_run():
    """Consecutive runs must get distinct, empty scratch dirs.

    This is the source-level fix for the `rm -rf /tmp/pscalib_us006_out` ritual:
    snapshot_calib() refuses (rightly) to overwrite a non-empty snapshot dir, so
    the scratch dir handed to it must never be one a previous run already filled.
    """
    first = make_out_dir("hyg05_idem")
    second = make_out_dir("hyg05_idem")
    try:
        assert first != second, (
            "two consecutive make_out_dir() calls returned the SAME path (%s) "
            "-- run N+1 would trip over run N's snapshot dir" % first)
        for d in (first, second):
            assert os.path.isdir(d), "make_out_dir did not create %s" % d
            assert os.listdir(d) == [], (
                "make_out_dir returned a NON-EMPTY dir (%s: %s) -- "
                "snapshot_calib would raise FileExistsError"
                % (d, os.listdir(d)))
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)
    print("[hyg05] make_out_dir() yields a fresh, empty scratch dir every call")


def test_us006_no_longer_hardcodes_a_reused_scratch_path():
    """test_geometry_us006.py must not reuse a fixed scratch path across runs.

    Checked against the AST, not the raw text: the file *documents* the old
    fixed path in a comment explaining why it is gone, and prose must not be
    able to fail (or pass) this check. Comments are absent from the AST, so what
    we assert on here is the code that actually executes.
    """
    src_path = os.path.join(_TESTS_DIR, "test_geometry_us006.py")
    with open(src_path) as fh:
        tree = ast.parse(fh.read(), filename=src_path)

    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert _OLD_FIXED_DIR not in literals, (
        "test_geometry_us006.py still has %r as a live string literal; if that "
        "is its scratch dir again, a second run dies with FileExistsError and "
        "the `rm -rf` ritual is back (HYG-05)" % _OLD_FIXED_DIR)

    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "make_out_dir" in calls, (
        "test_geometry_us006.py does not call make_out_dir() -- nothing "
        "guarantees it gets a fresh, empty scratch dir per run (HYG-05)")
    print("[hyg05] test_geometry_us006.py takes a fresh scratch dir per run")


def test_suite_is_idempotent_across_back_to_back_runs():
    """Running the runner twice in a row must give the same result, with no
    manual cleanup in between."""
    tmp = tempfile.mkdtemp(prefix="hyg05_twice_")
    try:
        probes = [
            _write_probe(tmp, "probe_pass.py", _PASS_PROBE),
            _write_probe(tmp, "probe_pass2.py", _PASS_PROBE),
        ]
        rc1, out1 = _run_runner(probes)
        rc2, out2 = _run_runner(probes)          # NO cleanup in between

        assert rc1 == 0, "first run failed. Output:\n" + _mask(out1)
        assert rc2 == 0, (
            "SECOND back-to-back run failed while the first passed -- the suite "
            "needs manual cleanup between runs (HYG-05). Output:\n"
            + _mask(out2))
        t1 = _SUMMARY_RE.search(out1)
        t2 = _SUMMARY_RE.search(out2)
        assert t1 and t2 and t1.groups() == t2.groups(), (
            "back-to-back runs disagree: %s vs %s"
            % (t1.groups() if t1 else None, t2.groups() if t2 else None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[hyg05] the runner is idempotent: two runs back to back, no cleanup")


def test_empty_suite_is_not_green():
    """A run that executes 0 tests must FAIL, not pass.

    If the manifest and tests/test_*.py were both empty, `0 passed, 0 failed,
    0 skipped` would otherwise satisfy the exit-0 condition -- a green light for
    a suite that proved nothing. The real manifest is non-empty (so its
    integrity check would reject an empty tests dir before we ever reach the
    guard), so we prove the guard EXECUTABLY on a sandbox copy of the runner
    whose MANIFEST has been emptied, pointed at an empty tests dir.
    """
    with open(_RUNNER) as fh:
        src = fh.read()
    # Empty the MANIFEST=( ... ) array in the copy (first block, up to a lone ')').
    sandbox_src, nsub = re.subn(r"MANIFEST=\(.*?\n\)", "MANIFEST=()", src,
                                count=1, flags=re.DOTALL)
    assert nsub == 1, "could not locate the MANIFEST=(...) block to empty it"

    tmp = tempfile.mkdtemp(prefix="hyg05_empty_")
    try:
        runner = os.path.join(tmp, "run_tests.sh")
        with open(runner, "w") as fh:
            fh.write(sandbox_src)
        os.makedirs(os.path.join(tmp, "tests"))          # empty: no test_*.py
        os.makedirs(os.path.join(tmp, "src"))            # silence PYTHONPATH warn
        proc = subprocess.run(["bash", runner],
                              cwd=tempfile.gettempdir(),
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        assert proc.returncode != 0, (
            "empty suite (0 tests) exited 0 -- a suite that ran nothing was "
            "called green (HYG-05). Output:\n" + _mask(out))
        assert "ran 0 tests" in out, (
            "empty-suite failure did not say 'ran 0 tests'. Output:\n"
            + _mask(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[hyg05] empty suite (0 tests) is RED, not green -- ran 0 tests => FAIL")


def main():
    print("=" * 72)
    print("HYG-05 acceptance: run_tests.sh is a meter (tally + skips + "
          "manifest + idempotency)")
    print("=" * 72)

    test_unlisted_skip_fails_the_run()
    test_bare_unrouted_skip_line_fails_the_run()
    test_summary_reports_an_explicit_skip_count()
    test_failing_test_is_counted_and_reddens_the_run()
    test_allowlisted_skip_is_forgiven()
    test_default_list_covers_every_test_on_disk()
    test_runner_reports_its_default_suite_and_it_matches_disk()
    test_scratch_dirs_are_fresh_every_run()
    test_us006_no_longer_hardcodes_a_reused_scratch_path()
    test_suite_is_idempotent_across_back_to_back_runs()
    test_empty_suite_is_not_green()

    print("\nALL HYG-05 RUNNER-HYGIENE CHECKS PASSED")


if __name__ == "__main__":
    main()
