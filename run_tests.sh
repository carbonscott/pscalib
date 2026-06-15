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
#   bash run_tests.sh [test_file ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # pscalib project root
SRC="$REPO/src"                                        # holds only pscalib/

# pscalib depends on the standalone numpy-only psdata package. Put its src/ on
# PYTHONPATH too (sibling repo) so `import psdata` resolves to the standalone
# package and the re-export shim resolves pscalib.
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

TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=(
    "$REPO/tests/test_calib_us000.py"
    "$REPO/tests/test_no_drift_us000.py"
    "$REPO/tests/test_webdb_us001.py"
    "$REPO/tests/test_validity_us002.py"
  )
fi

status=0
for t in "${TESTS[@]}"; do
  echo "### running $t"
  python3 "$t" || status=$?
done
exit $status
