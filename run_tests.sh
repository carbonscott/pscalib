#!/usr/bin/env bash
# Run the pscalib acceptance suite against the production psana on sdfiana025.
#
# The psana cross-check tests use a two-process oracle: psana (from psconda.sh,
# entered via PYTHONPATH) generates ground truth, and the numpy-only pscalib
# engine is compared against it. We expose pscalib AND its psdata dependency by
# prepending each project's src/ dir to PYTHONPATH; src/ holds ONLY the package,
# so `import pscalib` / `import psdata` resolve here while `import psana`
# resolves to the production env -- no shadowing.
#
# Usage (on sdfiana025):
#   source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
#   bash run_tests.sh                # the whole suite (the MANIFEST below)
#   bash run_tests.sh tests/foo.py   # just these files
#   bash run_tests.sh --list         # print the default suite and exit
#
# ---------------------------------------------------------------------------
# HYG-05: this runner is a METER, not a rubber stamp.
#
# It used to be five lines: loop, `python3 "$t" || status=$?`, exit. That gave
# no pass count, no fail count, and -- worst -- no SKIP count. The tests in this
# suite skip by printing a message and RETURNING, so they exit 0. A run in which
# psana was not importable and every single byte-exact oracle gate skipped
# itself still exited 0 and was recorded as "green". That is not a conformance
# suite, it is a green light with the bulb removed.
#
# So this runner now:
#   1. TALLIES.  N passed, M failed, S skipped -- explicitly, every run.
#   2. COUNTS SKIPS.  Tests emit `##SKIP## <name> :: <reason>` (tests/_skips.py);
#      the runner greps them out of each test's captured output, prints them,
#      and checks each name against tests/skips_allowed.txt. An unjustified skip
#      FAILS the run. A skip is not a pass. It ALSO catches skips announced the
#      OLD way -- a printed `[skip]`/`SKIPPED` line with no ##SKIP## marker --
#      and fails those too, so the protocol cannot be bypassed by not using it.
#      And it refuses to call an EMPTY suite (0 tests) green.
#   3. GUARDS THE MANIFEST.  The default list below is checked BOTH ways against
#      tests/test_*.py on disk: a test file that is not in the manifest fails the
#      run (that is how tests/test_no_drift_us000.py went unrun for months), and
#      a manifest entry with no file on disk fails the run too (that is how the
#      same file once got silently DELETED by a stray edit, with nobody noticing).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # pscalib project root
SRC="$REPO/src"                                        # holds only pscalib/
TESTS_DIR="$REPO/tests"
ALLOW_FILE="$TESTS_DIR/skips_allowed.txt"

# Must match tests/_skips.py:SKIP_MARKER.
SKIP_MARKER='##SKIP##'

# ---------------------------------------------------------------------------
# The MANIFEST: the authoritative, ordered default suite.
#
# Basenames only -- the integrity check below resolves them against $TESTS_DIR.
# It is checked bidirectionally against the files on disk, so it CANNOT silently
# drift out of sync in either direction. Add a tests/test_*.py file and you must
# add it here; delete one and the runner will tell you.
# ---------------------------------------------------------------------------
MANIFEST=(
  "test_no_drift_us000.py"
  "test_calib_us000.py"
  "test_webdb_us001.py"
  "test_validity_us002.py"
  "test_epix10ka_us004.py"
  "test_api_us005.py"
  "test_geometry_us006.py"
  "test_purity_us007.py"
  "test_epix10ka_trbit_us008.py"
  "test_cal07_provenance.py"
  "test_runner_hygiene_hyg05.py"
)

# ---------------------------------------------------------------------------
# Manifest integrity. Runs on EVERY invocation (it is instantaneous, and a
# guard you can bypass by passing an argument is not a guard).
# ---------------------------------------------------------------------------
check_manifest() {
  local errors=0 m f b found

  # (a) manifest -> disk: a listed test that no longer exists (deletion hazard).
  for m in "${MANIFEST[@]}"; do
    if [[ ! -f "$TESTS_DIR/$m" ]]; then
      echo "MANIFEST ERROR: '$m' is in run_tests.sh's MANIFEST but is MISSING from $TESTS_DIR/ -- was it deleted?" >&2
      errors=$((errors + 1))
    fi
  done

  # (b) disk -> manifest: a test file nobody wired up (the omission hazard).
  for f in "$TESTS_DIR"/test_*.py; do
    [[ -e "$f" ]] || continue
    b="$(basename "$f")"
    found=0
    for m in "${MANIFEST[@]}"; do
      if [[ "$m" == "$b" ]]; then
        found=1
        break
      fi
    done
    if [[ $found -eq 0 ]]; then
      echo "MANIFEST ERROR: '$b' exists in $TESTS_DIR/ but is NOT in run_tests.sh's MANIFEST -- it would never run. Add it." >&2
      errors=$((errors + 1))
    fi
  done

  if [[ $errors -ne 0 ]]; then
    echo "MANIFEST ERROR: the default test list does not match tests/test_*.py on disk ($errors problem(s))." >&2
    return 1
  fi
  return 0
}

check_manifest

# `--list`: report the resolved default suite and exit. Lets the HYG-05 hygiene
# test interrogate the manifest without needing psana or any data.
if [[ ${1:-} == "--list" ]]; then
  for m in "${MANIFEST[@]}"; do
    echo "$m"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# PYTHONPATH: pscalib + its standalone psdata dependency, PREPENDED (never
# replacing) so `import psana` still resolves to the production psconda env.
# ---------------------------------------------------------------------------
PSDATA_SRC="$(cd "$REPO/.." && pwd)/psdata/src"

PYPARTS="$SRC"
if [[ -d "$PSDATA_SRC" ]]; then
  PYPARTS="$SRC:$PSDATA_SRC"
else
  echo "WARNING: psdata src not found at $PSDATA_SRC -- pscalib depends on it" >&2
fi

if [[ -z "${PYTHONPATH:-}" ]]; then
  echo "WARNING: PYTHONPATH is empty -- did you source psconda.sh first?" >&2
  export PYTHONPATH="$PYPARTS"
else
  export PYTHONPATH="$PYPARTS:$PYTHONPATH"
fi

# ---------------------------------------------------------------------------
# Resolve what to run: explicit files from "$@", else the manifest.
# ---------------------------------------------------------------------------
TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=()
  for m in "${MANIFEST[@]}"; do
    TESTS+=("$TESTS_DIR/$m")
  done
fi

# ---------------------------------------------------------------------------
# Read the justified-skip allowlist: "<name> :: <justification>" records.
# ---------------------------------------------------------------------------
allow_names=()
allow_just=()
if [[ -f "$ALLOW_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    # trim leading/trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "$line" ]]; then continue; fi
    if [[ "${line:0:1}" == "#" ]]; then continue; fi
    if [[ "$line" != *" :: "* ]]; then
      echo "WARNING: malformed record in $ALLOW_FILE (want '<name> :: <justification>'): $line" >&2
      continue
    fi
    allow_names+=("${line%% :: *}")
    allow_just+=("${line#* :: }")
  done < "$ALLOW_FILE"
fi

# ---------------------------------------------------------------------------
# Run. Stream each test's output live AND capture it, so skips can be counted.
# ---------------------------------------------------------------------------
LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/pscalib_run_tests.XXXXXX")"
trap 'rm -rf "$LOGDIR"' EXIT

passed=0
failed=0
failed_names=()
n=0

for t in "${TESTS[@]}"; do
  n=$((n + 1))
  echo "### running $t"
  log="$(printf '%s/%03d_%s.log' "$LOGDIR" "$n" "$(basename "$t")")"
  # `tee` would hand us ITS exit status, not python's; pipefail would abort the
  # loop on the first failing test. Take python's status from PIPESTATUS[0] and
  # keep going, so one red test cannot hide the rest of the suite.
  set +e
  python3 "$t" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 0 ]]; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
    failed_names+=("$(basename "$t") (exit $rc)")
  fi
done

# ---------------------------------------------------------------------------
# Harvest the skip markers the tests emitted.
# ---------------------------------------------------------------------------
skip_names=()
skip_reasons=()
while IFS= read -r line; do
  if [[ -z "$line" ]]; then continue; fi
  rest="${line#*"$SKIP_MARKER"}"          # " <name> :: <reason>"
  rest="${rest#"${rest%%[![:space:]]*}"}" # ltrim
  if [[ "$rest" == *" :: "* ]]; then
    skip_names+=("${rest%% :: *}")
    skip_reasons+=("${rest#* :: }")
  else
    skip_names+=("$rest")
    skip_reasons+=("(no reason given)")
  fi
done < <(grep -hF -- "$SKIP_MARKER" "$LOGDIR"/*.log 2>/dev/null || true)

skipped=${#skip_names[@]}

# ---------------------------------------------------------------------------
# Bare (UNROUTED) skips: the marker protocol only meters skips that go through
# tests/_skips.py:skip(). A future author who writes the OLD idiom --
# `print("[skip] no psana"); return` -- exits 0 with no ##SKIP## marker and
# would score a silent PASS. That is the exact pre-fix bug (and this repo's own
# history shows discipline alone has failed here). So we also scan each test's
# captured log for a line that ANNOUNCES a skip the old way ("[skip]",
# "SKIPPED", "skipping") but carries NO marker, and treat it as unjustified.
#
# Lines that DO carry the marker are excluded: those are the real ##SKIP##
# lines, and our own prose that points a reader at them ("... SKIPPED -- see the
# ##SKIP## line above"). A green psana-present run prints none of these, so the
# scan is silent on a legitimately-green suite.
# ---------------------------------------------------------------------------
bare_skips=()
while IFS= read -r line; do
  [[ -n "$line" ]] && bare_skips+=("$line")
done < <(grep -hiE '\[skip\]|skipping|skipped' "$LOGDIR"/*.log 2>/dev/null \
           | grep -vF -- "$SKIP_MARKER" || true)
n_bare=${#bare_skips[@]}

# ---------------------------------------------------------------------------
# Empty-suite guard: "0 passed, 0 failed, 0 skipped" must NOT be green. A suite
# that ran nothing proves nothing.
# ---------------------------------------------------------------------------
if [[ $n -eq 0 ]]; then
  echo
  echo "======================================================================"
  echo "pscalib acceptance suite -- summary"
  echo "======================================================================"
  echo "0 passed, 0 failed, 0 skipped  (of 0 test file(s))"
  echo
  echo "RESULT: FAIL -- ran 0 tests (empty suite proves nothing)."
  exit 1
fi

# ---------------------------------------------------------------------------
# Summary + verdict.
# ---------------------------------------------------------------------------
echo
echo "======================================================================"
echo "pscalib acceptance suite -- summary"
echo "======================================================================"
echo "$passed passed, $failed failed, $skipped skipped  (of $n test file(s))"

if [[ ${#failed_names[@]} -gt 0 ]]; then
  echo
  echo "FAILED:"
  for f in "${failed_names[@]}"; do
    echo "  - $f"
  done
fi

unjustified=0
if [[ $skipped -gt 0 ]]; then
  echo
  echo "SKIPPED CHECKS ($skipped):"
  for i in "${!skip_names[@]}"; do
    name="${skip_names[$i]}"
    just=""
    if [[ ${#allow_names[@]} -gt 0 ]]; then
      for k in "${!allow_names[@]}"; do
        if [[ "${allow_names[$k]}" == "$name" ]]; then
          just="${allow_just[$k]}"
          break
        fi
      done
    fi
    echo "  - $name"
    echo "      reason: ${skip_reasons[$i]}"
    if [[ -n "$just" ]]; then
      echo "      justification: $just"
    else
      echo "      justification: NONE -- not in tests/skips_allowed.txt"
      echo "UNJUSTIFIED SKIP: $name -- a skip is not a pass (HYG-03/HYG-05)"
      unjustified=$((unjustified + 1))
    fi
  done
fi

if [[ $n_bare -gt 0 ]]; then
  echo
  echo "UNROUTED SKIPS ($n_bare) -- a skip announced the OLD way, bypassing skip():"
  for line in "${bare_skips[@]}"; do
    echo "  ! $line"
  done
  echo "UNJUSTIFIED SKIP: <unrouted> -- a skip is not a pass (HYG-03/HYG-05)."
  echo "  Route it through tests/_skips.py:skip(name, reason) so the runner can"
  echo "  count it and check it against tests/skips_allowed.txt -- a printed"
  echo "  '[skip]'/'SKIPPED' line with no ##SKIP## marker is an invisible skip."
fi

echo
if [[ $failed -ne 0 || $unjustified -ne 0 || $n_bare -ne 0 ]]; then
  if [[ $unjustified -ne 0 || $n_bare -ne 0 ]]; then
    echo "RESULT: FAIL -- $failed failed, $unjustified unjustified skip(s), $n_bare unrouted skip(s)."
    echo "A skipped oracle gate is NOT a passing oracle gate. Either fix the"
    echo "environment (psconda.sh sourced? PYTHONPATH prepended, not replaced?),"
    echo "route the skip through skip(name, reason), or justify it in $ALLOW_FILE."
  else
    echo "RESULT: FAIL -- $failed test file(s) failed."
  fi
  exit 1
fi

echo "RESULT: PASS -- $passed passed, 0 failed, $skipped skipped (all justified)."
exit 0
