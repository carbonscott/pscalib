#!/bin/bash
# FOUR-ARM CUBE PROOF -- one job, one exclusive milano node, 512 events.
#
#   1. before_orig   pscalib_incumbent (c5ce538) + UNMODIFIED harness
#   2. after_orig    pscalib_np8x                + UNMODIFIED harness
#   3. after_fast    pscalib_np8x                + bench_cube_calib_fast.py
#  (3b. after_fast2  pscalib_np8x                + bench_cube_calib_fast2.py
#       -- OPTIONAL, STRICTLY FAIL-SOFT, and it decides nothing; see EDIT 2)
#   4. before_again  pscalib_incumbent (c5ce538) + UNMODIFIED harness   [CONTROL,
#                    and LAST of the clocked arms]
#   5. STEP5 INSITU  un-clocked per-phase profile of accumulate_slice() at
#                    nbins=10, both harnesses, candidate pscalib. Clocks nothing.
#
# The arms are separated so the two halves of the speedup are attributable:
#   after_orig / before_orig   = what the pscalib SOURCE change bought
#   after_fast / after_orig    = what the HARNESS patch (prefetch overlap,
#                                reused buffers, all-finite fast path) bought
#   before_orig / after_fast   = the campaign number, threshold >= 8.0x
#
# WHY A FOURTH ARM.  512 events x 33.55 MB = 17.2 GB, and the node has 503 GiB.
# So after arm 1 has read the whole event set, arms 2-4 read it from PAGE CACHE.
# Every one of those cache hits makes the LATER arms look faster, and the later
# arms are exactly the ones whose speedup this job is trying to establish: the
# cache bias points ENTIRELY toward passing. before_again re-runs the incumbent
# LAST, with the cache as warm as it will ever be, so the bias is measured
# instead of argued about. CACHE_CONTROL = before_again / before_orig (evt/s):
# 1.00 means the cache bought the later arms nothing; a large value means the
# headline ratio is inflated by that factor and the number must be re-taken.
# It is printed and FLAGGED outside 0.90..1.10, but it does NOT by itself fail
# the job -- deciding that a flagged number is unusable is a human call, and
# suppressing the printed ratio would hide the very evidence needed to make it.
#
# ==========================================================================
# WHAT THIS SCRIPT KEEPS FROM _cand/bench_cube_candidate.sbatch (all of the
# hard-won hardening; do not remove any of it):
#   * EXP_CAND_SHA is REQUIRED, no default. Unset => exit 63 before the clock.
#     (All four iteration-1 null runs left it unset and printed
#     LIB_PSCALIB_SHA_UNPINNED -- precisely the check that catches a dirty or
#     wrong candidate worktree.)
#   * Per arm, PRE and POST: pscalib SHA + dirty-file count + src md5 (md5 of the
#     md5s of every .py under src, RELATIVE paths so the fingerprint is
#     location-independent), re-asserted AFTER the arm finishes. Another worker
#     editing a worktree mid-benchmark would otherwise be invisible; the
#     candidate worktree went from clean to 4 modified files inside 9 minutes on
#     2026-08-05.
#   * The psana release SHA and the psdata SHA are hard-asserted.
#   * A python provenance probe per arm: the psdata/pscalib actually IMPORTED
#     must come from this arm's PYTHONPATH entries, and psana from the release.
#   * Per-arm tee'd logs, so the RESULT lines can be parsed afterwards.
#   * The embedded BITGATE byte-exactness proof, run AFTER the clock so it
#     cannot perturb it, in both numba-active and numba-blocked modes. The
#     harness's own GATE PASS / PURITY=PASS lines do NOT test byte-exactness:
#     run_gate() checks bin-key provenance, "calib is real", parallel==serial,
#     zero rescan and import purity, and PURITY=PASS only asserts
#     isfinite(cal).any(). A candidate that dropped masking entirely would print
#     all three. (On the np8x tree numba is not merely unused, it is deleted --
#     so --block-numba there is a proof that numpy ALONE suffices to import and
#     calibrate, which is this campaign's central constraint.)
#
# WHAT THIS SCRIPT ADDS
#   * Four arms, and the harness FILE each arm loaded is md5-asserted: the three
#     "orig" arms must show a29e67e5c453fb5cd31e9184fe0e3f7c (the shared,
#     unmodified harness -- which this job never writes to), after_fast must show
#     the patched md5.
#   * dirty==0 is now asserted on EVERY arm, not just the incumbent. A clean
#     worktree at a pinned SHA IS its content, so SHA+dirty together pin the
#     bytes; the src md5 is then a printed cross-check and a mid-arm tripwire.
#   * OUTGATE on the two np8x arms: the new out= buffer path must be bit-exact
#     against the frozen reference frames on THIS node, not just on the login
#     node where it was developed.
#   * One shared constants npz, written by the first arm and read by the other
#     three, so all four arms provably calibrate with the SAME constant bytes
#     (and only one arm depends on the web DB being up).
#
# WHAT THIS SCRIPT DELIBERATELY DOES **NOT** INHERIT
#   * The predecessor's MB_per_s arm-symmetry reject rule (its exit 96). That
#     rule voided a job when the two arms' MB_per_s differed by more than 15%.
#     But MB_per_s = nbytes/wall with nbytes FIXED (the same 512 events, the
#     same 17.2 GB, read by every arm), so MB_per_s is just 1/wall in disguise
#     and the rule is a WALL-TIME SYMMETRY test: any genuine speedup above
#     ~1.18x fails it by construction. It made all three predecessor proof jobs
#     exit rc=96 while every single arm exited rc=0. **rc=96 from the old rule
#     carries ZERO information about those runs.** Do not reintroduce it and do
#     not read anything into the predecessor's 96s.
#   * The replacement symmetry check is on BYTES READ, not on bandwidth: every
#     arm reads the same 512 events, so their rchar totals must agree, and that
#     statement is completely independent of how fast each arm ran. It is
#     reconstructed per arm as MB_per_s x wall (the RESULT line carries an
#     integer MB_per_s, so this is ~1% coarse -- the thresholds below are set
#     far outside that). A >15% spread is WARNed; a >50% spread exits 97,
#     because at that point the arms did not read the same data and no ratio
#     between them means anything.
#
# ==========================================================================
# STAGE-4 EDITS (this file is proof4arm.sbatch, derived from the NEVER-RUN
# proof3arm.sbatch; every change is listed here so the diff is auditable):
#   1. --time 03:00:00 -> 06:00:00.  A job that dies at the walltime destroys
#      the whole iteration; the arithmetic below now budgets for BOTH incumbent
#      arms being hit by the predecessor's 5.7x distortion at once.
#   2. The OPTIONAL fifth arm after_fast2 is now STRICTLY FAIL-SOFT: its
#      CUBE_ARM_EXIT rc can no longer set rc_all and its BITGATE/OUTGATE rcs can
#      no longer set gate_bad.  It was already fail-soft on a bad md5, but a
#      CRASH inside it would have made the whole job exit non-zero even with all
#      four required arms green -- exactly the predecessor's rc=96 failure mode
#      in a new costume.  The four required arms decide this job; the fifth is a
#      free extra measurement and nothing more.  It still runs BEFORE
#      before_again so that the CONTROL arm is LAST, as specified.
#   3. NEW un-clocked STEP 5 at the very end: insitu_probe.py, which imports the
#      harness (patched AND unmodified) and profiles its OWN accumulate_slice()
#      over 64 real events at nbins=10, printing
#      INSITU ... read_ms= stack_ms= calib_ms= accum_ms= total_ms= overlap_f=.
#      This exists because a skeptic quarantined the 43.94 ms accumulate figure:
#      it was taken with ONE bin while the harness runs nbins=10.  The probe
#      also reruns the same events with every bin forced to 0, so the bin-count
#      effect is MEASURED rather than argued about.  It runs after everything
#      that is clocked, so it cannot perturb a single number above.
#   4. Post-arm re-assert now also re-checks the SHA against EXP_CAND_SHA /
#      EXP_INCUMBENT explicitly, not only against the arm's own pre value.
#   5. SLURM_JOB_ID falls back to $$ so a `bash proof4arm.sbatch` dry run does
#      not create a directory called armlogs- .
#   6. RE-CONFIRMED REMOVED: there is NO MB_per_s arm-symmetry reject rule
#      anywhere in this file (see "WHAT THIS SCRIPT DELIBERATELY DOES NOT
#      INHERIT" above).  The only symmetry check is BYTES_SYMMETRY, on
#      MB_per_s x wall == bytes read, which is speed-independent by
#      construction.  Verify with:  grep -n 'MB_per_s' proof4arm.sbatch
#
# STAGE-4b EDITS (this iteration, after the 8-event smoke job 34323653 ran the
# whole file end to end on sdfmilan241 and exited 90 = "clean, below target"):
#   7. Item 6 above was NOT true as written: BYTES_SYMMETRY still contained
#      `fail = 97`, so a spread > 1.50 WOULD have voided the job. That exit is
#      now removed and the check is report-only. It is the last auto-void on a
#      symmetry criterion; nothing in this file rejects on symmetry any more.
#   8. squeue -u is snapshotted at job START and job END. The node is
#      exclusive, the FILESYSTEM is not, and the user's own `catalog-index`
#      (34321052, 32 CPUs on sdfmilan104) plus several `no-shell` allocations
#      share it. If this job lands on sdfmilan104 the record must say so.
#   9. STEP 5 now profiles the level-2 harness too, with BENCH_FAST_BATCH
#      exported for that invocation only. The level-2 arm is in the job; its
#      accumulate decomposition is the single most useful thing step 5 can hand
#      the next iteration, and step 5 is advisory so it cannot cost anything.
# ==========================================================================
#
# WALLTIME ARITHMETIC (--time below)
#   Per-arm fixed cost, paid by EVERY arm: index build ~30 s + constants
#   (1.2 GB npz: ~60 s to fetch+write on arm 1, ~20 s to load on arms 2-5) +
#   ray.init ~10 s  =~ 60-100 s.
#   before_orig  512 ev x 0.6453 s/ev = 330 s at workers=1
#                + ~62 s at workers=16 (20.7 s ideal x ~3 for Ray/IO overhead)
#                + ~120 s gate (48-event serial oracle at 0.645 s/ev = 31 s, the
#                  parallel re-cube of the same 48, the purity subprocess, the
#                  all-event bin-key check)
#                + ~80 s of BITGATE (2 modes x 8 frames x 0.556 s/frame plus
#                  two 536 MB reference loads)
#                + ~100 s fixed                                 =~ 690 s
#   before_again identical minus the npz write                  =~ 630 s
#   after_orig   ~175 ms/ev (read 38.7 + stack 10.4 + calib 52.9 + accum 44-72;
#                the accumulate term is the QUARANTINED one, taken at its
#                pessimistic end here on purpose)
#                -> 90 s at w=1, ~25 s at w=16, ~60 s gate,
#                   ~70 s bitgate, ~50 s outgate, ~60 s fixed   =~ 355 s
#   after_fast   <= 80.7 ms/ev target -> 41 s at w=1, ~15 s at
#                w=16, ~50 s gate, ~70 s bitgate, ~50 s outgate,
#                ~60 s fixed                                    =~ 290 s
#   after_fast2  same shape as after_fast                       =~ 290 s
#   STEP 5 INSITU  2 harnesses x (index 30 + constants 20 + warmup + 64 ev +
#                64 ev one-bin + 256 ev + 512 ev)               =~ 350 s
#   TOTAL nominal  690+630+355+290+290+350                     =~ 2605 s = 44 min
#   DISTORTION ENVELOPE.  The predecessor saw ONE arm take 1882 s where 330 s
#   was expected (5.7x).  Assume BOTH incumbent arms are hit at once:
#   2 x (330 x 5.7 - 330) = +3140 s.  Assume the three candidate arms are hit
#   too: 3 x (90 x 5.7 - 90) = +1270 s.  2505 + 3140 + 1270 = 6915 s = 115 min.
#   Doubling that for a slow node / a cold cache / queue-time-adjacent
#   weirdness gives 230 min.  --time=06:00:00 = 360 min covers it with 130 min
#   of headroom.  Nothing about this job is O(walltime): an unused hour costs
#   the campaign nothing, an hour short costs it the entire iteration.
#SBATCH --partition=milano
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/sdf/data/lcls/ds/prj/prjcwang31/results/_cand2/proof-%x-%j.out
set -o pipefail

SW=/sdf/data/lcls/ds/prj/prjcwang31/results/software
CWT=$SW/_cand_worktrees
RUNS=/sdf/data/lcls/ds/prj/prjcwang31/results/_bench_runs
CAND=/sdf/data/lcls/ds/prj/prjcwang31/results/_cand
CAND2=/sdf/data/lcls/ds/prj/prjcwang31/results/_cand2

HARNESS_ORIG=$RUNS/harness/bench_cube_calib.py          # SHARED -- never written
HARNESS_FAST=$CAND2/bench_cube_calib_fast.py            # produced by patch_harness.py
# ORCHESTRATOR ADDITION -- a FIFTH arm. bench_cube_calib_fast2.py is the
# INSERTION-ONLY level-2 patch (232 added lines, 0 removed) that adds the
# batched float64 accumulate on top of level 1, and it is already proved
# bit-inert at K=1/4/64 over five phases (real, nanlane, mixlane, interleave,
# interleave_nan) in _cand2/cube_inertness2.log. The projection for after_fast
# straddles the 8.0 threshold on the accumulate term alone, so measuring both
# levels in ONE job is what avoids a second multi-hour turnaround. This arm is
# FAIL-SOFT: if the file is absent or its md5 is wrong, it is SKIPPED with a
# loud message and the four required arms proceed untouched.
HARNESS_FAST2=$CAND2/bench_cube_calib_fast2.py
BITGATE=$CAND/bitgate_fixture.py
OUTGATE=$CAND2/outgate_fixture.py
INSITU=$CAND2/insitu_probe.py                           # STEP 5, un-clocked

EXP_H_ORIG=a29e67e5c453fb5cd31e9184fe0e3f7c
# The patched harness's md5, as produced by _cand2/patch_harness.py from the
# harness above. Overridable so a regenerated patch can be pinned by the caller,
# but NEVER skipped: the whole point of the after_fast arm is that we know which
# bytes ran.
EXP_H_FAST="${EXP_HARNESS_FAST_MD5:-cf794d11ba8689ca46cd8bc0581e8f9c}"
EXP_H_FAST2="${EXP_HARNESS_FAST2_MD5:-1558f52bcf9f6745233b14c48d66cf9b}"
# Events buffered before the grouped float64 flush. K x 67 MB of float32 ring
# buffers: 4.3 GB at K=64, against 503 GiB on the node.
FAST2_BATCH="${BENCH_FAST_BATCH_K:-64}"

PSANA_REL=/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_070626
EXP_PSANA=7967fd21743946c72bc77a28d88165b6557b68e5
EXP_PSDATA=15caa3a1695aabc5d148a2efda4ec8aff9fbd582
EXP_INCUMBENT=c5ce538981c62b975d0d5e672070c7ec6afe0fb5

PSD=$SW/psdata                      # SAME live psdata for every arm
PSC_BEFORE=$CWT/pscalib_incumbent
PSC_AFTER="${BENCH_PSC_AFTER:-$CWT/pscalib_np8x}"

EVENTS="${BENCH_EVENTS:-512}"
WORKERS="${BENCH_WORKERS_ARGV:-1 16}"
BINS="${BENCH_BINS:-10}"
BINVAR="${BENCH_BINVAR:-epics:pulse_energy_diode_txi}"
RATIO_TARGET="${RATIO_TARGET:-8.0}"

# LANDMINE, found the hard way by job 34300338: the harness ALSO reads
# BENCH_WORKERS from the environment, AT IMPORT TIME, and splits it on COMMA
# (bench_cube_calib.py:204). sbatch propagates the caller's environment, so an
# inherited BENCH_WORKERS='1 16' makes the harness die with
#   ValueError: invalid literal for int() with base 10: '1 16'
# before argparse ever runs. --workers on argv is what this script controls, so
# the env var is removed from the job environment entirely. (This script's own
# knob is BENCH_WORKERS_ARGV, deliberately a different name.)
unset BENCH_WORKERS

[ -n "${BENCH_ALLOW_NONBATCH:-}" ] && { echo 'FATAL: BENCH_ALLOW_NONBATCH set'; exit 65; }
if [ -z "${EXP_CAND_SHA:-}" ]; then
  echo 'FATAL: EXP_CAND_SHA is REQUIRED and is empty/unset -- refusing to start.'
  echo '  The candidate arms SHA must be pinned by the caller. All four'
  echo '  iteration-1 null runs left it unset and printed LIB_PSCALIB_SHA_UNPINNED,'
  echo '  which is exactly the check that catches a dirty or wrong worktree.'
  echo "  Set it e.g.:  EXP_CAND_SHA=\$(git -C $PSC_AFTER rev-parse HEAD)"
  exit 63
fi
for f in "$HARNESS_ORIG" "$HARNESS_FAST" "$BITGATE" "$OUTGATE" "$INSITU"; do
  [ -r "$f" ] || { echo "FATAL: missing/unreadable $f"; exit 67; }
done
: "${SLURM_JOB_ID:=dryrun$$}"
LOGDIR=$CAND2/armlogs-$SLURM_JOB_ID
mkdir -p $LOGDIR || { echo "FATAL: cannot mkdir $LOGDIR"; exit 67; }
NPZ=$LOGDIR/cube_constants.npz

# Fingerprint of the python sources actually importable from $1. The paths fed
# to md5sum are RELATIVE to $1 on purpose: md5sum prints the path next to the
# digest, so hashing absolute paths would make two byte-identical trees in
# different directories fingerprint DIFFERENTLY. Relative paths keep the
# filenames in the fingerprint -- a rename is still caught -- while making it
# location-independent.
srcmd5() { ( cd "$1" && find src -name '*.py' -type f -print0 | sort -z | xargs -0 md5sum ) | md5sum | cut -d' ' -f1; }
filemd5() { md5sum "$1" | cut -d' ' -f1; }

H_ORIG_MD5=$(filemd5 $HARNESS_ORIG)
H_FAST_MD5=$(filemd5 $HARNESS_FAST)
RUN_FAST2=0
if [ -r "$HARNESS_FAST2" ]; then
  H_FAST2_MD5=$(filemd5 $HARNESS_FAST2)
  if [ "$H_FAST2_MD5" = "$EXP_H_FAST2" ]; then
    RUN_FAST2=1
    echo "HARNESS_FAST2   $HARNESS_FAST2  md5=$H_FAST2_MD5  ENABLED K=$FAST2_BATCH"
  else
    echo "FAST2_SKIPPED md5 mismatch got=$H_FAST2_MD5 want=$EXP_H_FAST2 -- the"
    echo "  optional fifth arm is SKIPPED; the four required arms are unaffected."
  fi
else
  echo "FAST2_SKIPPED $HARNESS_FAST2 not readable -- optional fifth arm skipped."
fi
echo "=== CUBE-PROOF-4ARM node=$(hostname) job=$SLURM_JOB_ID events=$EVENTS workers=$WORKERS bins=$BINS binvar=$BINVAR $(date) ==="
echo "PSC_BEFORE      $PSC_BEFORE"
echo "PSC_AFTER       $PSC_AFTER   (EXP_CAND_SHA=$EXP_CAND_SHA)"
echo "HARNESS_ORIG    $HARNESS_ORIG  md5=$H_ORIG_MD5  (expect $EXP_H_ORIG)"
echo "HARNESS_FAST    $HARNESS_FAST  md5=$H_FAST_MD5  (expect $EXP_H_FAST)"
echo "ARMLOGS         $LOGDIR"
echo "CONSTANTS_NPZ   $NPZ  (written by arm 1, read by arms 2-4: all four arms"
echo "                provably calibrate with the SAME constant bytes)"
echo "INSITU_PROBE    $INSITU  md5=$(filemd5 $INSITU)"
# STAGE-4b EDIT (8). The NODE is exclusive; the FILESYSTEM is not. The user's
# own catalog-index (34321052, 32 CPUs on sdfmilan104) and several no-shell
# allocations share /sdf/data/lcls, and the predecessor campaign lost three
# proof runs to an I/O storm from concurrent xtc readers. Snapshot what else of
# ours is on the cluster, at the start and again at the end, so the record can
# say whether anything of ours was reading alongside this job -- and, in
# particular, whether this job landed on the same node as catalog-index.
echo "--- SQUEUE_AT_START ---"
squeue -u "$(whoami)" -o "%.10i %.14j %.9T %.14R %.6D %.5C %.11M %.11l" 2>&1
echo "--- end SQUEUE_AT_START ---"
echo "NODE_AT_START $(hostname)"
# Assert the harness files up front, BEFORE the clock, so a stale patched
# harness cannot burn three hours of node time.
[ "$H_ORIG_MD5" = "$EXP_H_ORIG" ] || { echo "FATAL: the SHARED harness is not the pinned file (got $H_ORIG_MD5 want $EXP_H_ORIG)"; exit 99; }
[ "$H_FAST_MD5" = "$EXP_H_FAST" ] || { echo "FATAL: the PATCHED harness is not the pinned file (got $H_FAST_MD5 want $EXP_H_FAST) -- re-run patch_harness.py or pass EXP_HARNESS_FAST_MD5"; exit 99; }
echo "HARNESS_MD5S_ASSERTED ok"

source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
[ -n "${PYTHONPATH:-}" ] || { echo 'FATAL: PYTHONPATH empty after psconda'; exit 66; }
BASE_PP=$PYTHONPATH
PSANA_SHA=$(git -c safe.directory='*' -C $PSANA_REL rev-parse HEAD)
echo "PSANA_RELEASE_SHA: $PSANA_SHA"
[ "$PSANA_SHA" = "$EXP_PSANA" ] || { echo 'FATAL: psana SHA mismatch/empty'; exit 94; }

rc_all=0
gate_bad=0
ARMS='before_orig after_orig after_fast before_again'
# after_fast2 runs immediately after after_fast so the two patch levels sit
# adjacent in time and see the most similar page-cache state.
[ "$RUN_FAST2" = "1" ] && ARMS='before_orig after_orig after_fast after_fast2 before_again'
echo "ARM_ORDER $ARMS"

for ARM in $ARMS; do
  case "$ARM" in
    before_orig|before_again) PSC=$PSC_BEFORE; HRN=$HARNESS_ORIG; EXPH=$EXP_H_ORIG; KIND=incumbent ;;
    after_orig)               PSC=$PSC_AFTER;  HRN=$HARNESS_ORIG; EXPH=$EXP_H_ORIG; KIND=candidate ;;
    after_fast)               PSC=$PSC_AFTER;  HRN=$HARNESS_FAST; EXPH=$EXP_H_FAST; KIND=candidate ;;
    after_fast2)              PSC=$PSC_AFTER;  HRN=$HARNESS_FAST2; EXPH=$EXP_H_FAST2; KIND=candidate ;;
    *) echo "FATAL: unknown arm $ARM"; exit 64 ;;
  esac
  echo "########## CUBE_ARM=$ARM kind=$KIND node=$(hostname) job=$SLURM_JOB_ID $(date) ##########"
  export PYTHONPATH=$PSC/src:$PSD/src:$BASE_PP
  export PYTHONDONTWRITEBYTECODE=1        # never write .pyc into a read-only worktree
  # BENCH_FAST_BATCH is read only by the level-2 harness and defaults to 0
  # (OFF) there, but it is unset explicitly on every other arm so an inherited
  # value can never silently change what a level-1 or unmodified arm does.
  if [ "$ARM" = after_fast2 ]; then export BENCH_FAST_BATCH=$FAST2_BATCH;
  else unset BENCH_FAST_BATCH; fi
  echo "ARM_ENV arm=$ARM BENCH_FAST_BATCH=${BENCH_FAST_BATCH:-<unset>}"

  GOT=$(git -C $PSD rev-parse HEAD); echo "LIB_PSDATA_SHA:  $GOT"
  [ "$GOT" = "$EXP_PSDATA" ] || { echo 'FATAL: psdata SHA mismatch'; exit 91; }
  GOT=$(git -C $PSC rev-parse HEAD)
  DIRTY=$(git -C $PSC status --porcelain | wc -l)
  SRCMD5=$(srcmd5 $PSC)
  HMD5=$(filemd5 $HRN)
  echo "LIB_PSCALIB_SHA: $GOT"
  echo "LIB_PSCALIB_DIRTY_FILES: $DIRTY"
  echo "LIB_PSCALIB_SRC_MD5: $SRCMD5"
  echo "ARM_HARNESS arm=$ARM file=$HRN md5=$HMD5 expect=$EXPH"
  [ "$HMD5" = "$EXPH" ] || { echo "FATAL: arm=$ARM loaded harness md5=$HMD5, expected $EXPH"; exit 99; }
  # A SHA pins HEAD, NOT the working tree: a dirty worktree at the right HEAD
  # passes a bare rev-parse assert. dirty==0 is therefore asserted on EVERY arm,
  # which together with the SHA pins the actual bytes that get imported.
  [ "$DIRTY" = "0" ] || { echo "FATAL: arm=$ARM worktree $PSC is DIRTY ($DIRTY files) -- the SHA does not describe what would run"; exit 92; }
  if [ "$KIND" = incumbent ]; then
    [ "$GOT" = "$EXP_INCUMBENT" ] || { echo "FATAL: incumbent pscalib SHA mismatch got=$GOT want=$EXP_INCUMBENT"; exit 92; }
    echo "LIB_PSCALIB_SHA_ASSERTED: $GOT (incumbent, clean)"
  else
    [ "$GOT" = "$EXP_CAND_SHA" ] || { echo "FATAL: candidate pscalib SHA mismatch got=$GOT want=$EXP_CAND_SHA"; exit 92; }
    echo "LIB_PSCALIB_SHA_ASSERTED: $GOT (candidate, EXP_CAND_SHA, clean)"
  fi
  ARM_SHA0=$GOT; ARM_DIRTY0=$DIRTY; ARM_MD50=$SRCMD5

  python - <<'PY' || exit 93
import os
import psdata, pscalib, psana
w = os.environ['PYTHONPATH'].split(':')
print('LOADED psdata :', psdata.__file__)
print('LOADED pscalib:', pscalib.__file__)
assert psdata.__file__.startswith(w[1]), 'WRONG psdata!'
assert pscalib.__file__.startswith(w[0]), 'WRONG pscalib!'
assert '/rel/lcls2_' in psana.__file__, 'psana not from release!'
print('PROVENANCE OK')
PY

  # jungfrau, real .calib(), real EPICS binning axis (r51 has NO commanded scan;
  # pulse_energy_diode_txi is the most-resolved var that actually varies).
  # stdout is tee'd per arm so the RESULT lines can be parsed for the ratios.
  python $HRN --det jungfrau --workers $WORKERS --events $EVENTS --bins $BINS \
              --bin-var $BINVAR --constants-npz $NPZ 2>&1 | tee $LOGDIR/arm_$ARM.log
  rc=${PIPESTATUS[0]}; echo "CUBE_ARM_EXIT arm=$ARM rc=$rc"
  # STAGE-4 EDIT (2): the OPTIONAL fifth arm can never fail this job. The four
  # required arms decide it; a crash in the free extra measurement must not
  # turn a green four-arm proof into a red job (that is precisely how the
  # predecessor's rc=96 destroyed three otherwise-clean runs).
  if [ "$ARM" = after_fast2 ]; then
    [ $rc -ne 0 ] && echo "FAST2_NONFATAL arm=$ARM rc=$rc -- optional arm, NOT counted against the job"
  else
    [ $rc -ne 0 ] && rc_all=$rc
  fi

  # POST-ARM re-assert: the tree that was benchmarked must be the tree that was
  # asserted. Catches a concurrent edit/commit landing mid-arm.
  SHA1=$(git -C $PSC rev-parse HEAD)
  DIRTY1=$(git -C $PSC status --porcelain | wc -l)
  MD51=$(srcmd5 $PSC)
  HMD51=$(filemd5 $HRN)
  echo "LIB_PSCALIB_POSTARM arm=$ARM sha=$SHA1 dirty=$DIRTY1 src_md5=$MD51 harness_md5=$HMD51"
  if [ "$SHA1" != "$ARM_SHA0" ] || [ "$DIRTY1" != "$ARM_DIRTY0" ] || [ "$MD51" != "$ARM_MD50" ] || [ "$HMD51" != "$EXPH" ]; then
    echo "FATAL: pscalib worktree or harness CHANGED DURING arm=$ARM (before: sha=$ARM_SHA0 dirty=$ARM_DIRTY0 md5=$ARM_MD50 harness=$EXPH)"
    exit 92
  fi
  # STAGE-4 EDIT (4): re-assert against the PINNED constant too, not only
  # against this arm's own pre-value. Comparing an arm to itself cannot catch a
  # tree that was already wrong when the arm started -- the pre-check catches
  # that, and this makes the post-check say so in its own words.
  if [ "$KIND" = incumbent ]; then
    [ "$SHA1" = "$EXP_INCUMBENT" ] || { echo "FATAL: POSTARM incumbent SHA $SHA1 != $EXP_INCUMBENT"; exit 92; }
  else
    [ "$SHA1" = "$EXP_CAND_SHA" ] || { echo "FATAL: POSTARM candidate SHA $SHA1 != EXP_CAND_SHA $EXP_CAND_SHA"; exit 92; }
  fi
  [ "$DIRTY1" = "0" ] || { echo "FATAL: POSTARM arm=$ARM dirty=$DIRTY1"; exit 92; }
  echo "LIB_PSCALIB_POSTARM_STABLE arm=$ARM sha=$SHA1 dirty=$DIRTY1 src_md5=$MD51 harness_md5=$HMD51 (re-asserted against the pinned SHA)"

  # ---- EMBEDDED BYTE-EXACTNESS PROOFS -- after the clock, so they cannot
  # perturb it. The harness's GATE PASS line does not test byte-exactness.
  BGARM=before; [ "$KIND" = candidate ] && BGARM=after
  # STAGE-4 EDIT (2): the optional arm's gates are advisory too.
  OPTARM=0; [ "$ARM" = after_fast2 ] && OPTARM=1
  echo "---------- BITGATE arm=$ARM (numba-active) ----------"
  python $BITGATE --arm $BGARM --expect-prefix $PSC --label "job=$SLURM_JOB_ID arm=$ARM"
  bg1=$?; echo "BITGATE_EXIT arm=$ARM mode=numba_active rc=$bg1"
  [ $bg1 -ne 0 ] && [ $OPTARM -eq 0 ] && gate_bad=1
  echo "---------- BITGATE arm=$ARM (numba-BLOCKED: numpy ALONE must suffice) ----------"
  python $BITGATE --arm $BGARM --block-numba --expect-prefix $PSC --label "job=$SLURM_JOB_ID arm=$ARM"
  bg2=$?; echo "BITGATE_EXIT arm=$ARM mode=numba_blocked rc=$bg2"
  [ $bg2 -ne 0 ] && [ $OPTARM -eq 0 ] && gate_bad=1

  if [ "$KIND" = candidate ]; then
    # The out= buffer path is new in the candidate and is what the patched
    # harness relies on; prove it bit-exact on THIS node, not just where it was
    # written. (The incumbent has no out=, so this gate is candidate-only.)
    echo "---------- OUTGATE arm=$ARM (out= buffer must be bit-identical) ----------"
    python $OUTGATE
    og=$?; echo "OUTGATE_EXIT arm=$ARM rc=$og"
    [ $og -ne 0 ] && [ $OPTARM -eq 0 ] && gate_bad=1
  fi
done

# ==========================================================================
# THE RATIOS -- computed inside the job from the parsed RESULT lines.
# ==========================================================================
echo "########## PROOF_VERDICT job=$SLURM_JOB_ID ##########"
LOGDIR=$LOGDIR RATIO_TARGET=$RATIO_TARGET python - <<'PYVERDICT'
import os
import re
import sys

LOGDIR = os.environ["LOGDIR"]
TARGET = float(os.environ.get("RATIO_TARGET", "8.0"))
ARMS = ["before_orig", "after_orig", "after_fast", "before_again"]
# after_fast2 is OPTIONAL: it must never be able to fail the verdict, because
# the campaign threshold is defined on the four required arms.
OPT_ARMS = ["after_fast2"]

# RESULT det=... workers=1 wall=333.00s evt_per_s=1.54 ... MB_per_s=51
#        partition_invariant=1 VERDICT=PASS
KV = re.compile(r"(\w+)=([^\s]+)")


def parse(arm):
    path = os.path.join(LOGDIR, "arm_%s.log" % arm)
    rows = {}
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("RESULT "):
                    continue
                d = dict(KV.findall(line.strip()))
                w = int(d["workers"])
                rows[w] = dict(
                    wall=float(d["wall"].rstrip("s")),
                    rate=float(d["evt_per_s"]),
                    e2e=float(d["end_to_end_evt_per_s"]),
                    mbs=float(d["MB_per_s"]),
                    events=int(d["events"]),
                    pinv=int(d["partition_invariant"]),
                    verdict=d["VERDICT"],
                )
    except OSError as e:
        print("PARSE_FAIL arm=%s %s" % (arm, e))
    return rows


R = {a: parse(a) for a in ARMS + OPT_ARMS}
ALL_ARMS = ARMS + [a for a in OPT_ARMS if R.get(a)]
missing = [a for a in ARMS if not R[a]]
if missing:
    print("PROOF_VERDICT FAIL: no RESULT lines for arm(s) %s -- an arm did not "
          "reach the timing section (its own CUBE_ARM_EXIT rc and GATE lines "
          "above say why). No ratio is printed, because a ratio against a "
          "missing arm is not a number." % missing)
    sys.exit(96)

print("%-13s %7s %10s %10s %13s %10s %12s %6s %8s"
      % ("arm", "workers", "wall_s", "evt_per_s", "ms_per_event", "MB_per_s",
         "bytes_read_GB", "p_inv", "verdict"))
for a in ALL_ARMS:
    for w in sorted(R[a]):
        r = R[a][w]
        ms = 1000.0 / r["rate"] if r["rate"] else float("nan")
        gb = r["mbs"] * r["wall"] / 1e3          # MB/s * s = MB -> GB
        r["ms"] = ms
        r["gb"] = gb
        print("%-13s %7d %10.2f %10.2f %13.2f %10.0f %12.2f %6d %8s"
              % (a, w, r["wall"], r["rate"], ms, r["mbs"], gb, r["pinv"],
                 r["verdict"]))
        print("ARM_ROW arm=%s workers=%d wall=%.2f evt_per_s=%.4f "
              "ms_per_event=%.3f MB_per_s=%.0f bytes_read_GB=%.3f events=%d "
              "partition_invariant=%d verdict=%s"
              % (a, w, r["wall"], r["rate"], ms, r["mbs"], gb, r["events"],
                 r["pinv"], r["verdict"]))

fail = 0
W = 1
if any(W not in R[a] for a in ARMS):
    print("PROOF_VERDICT FAIL: workers=1 row missing from some arm; the "
          "campaign target is defined at workers=1.")
    sys.exit(96)

bo, ao, af, ba = (R[a][W] for a in ARMS)

# RATIO_MAIN is a TIME ratio (ms/event before / ms/event after): >1 means
# faster, and 8.0 is the campaign threshold (645.3 -> 80.7 ms/event).
ratio_main = bo["ms"] / af["ms"]
# The two legs are RATE ratios (evt/s after / evt/s before): also >1 == faster.
# They multiply exactly to RATIO_MAIN, which is printed as a self-check.
ratio_pscalib = ao["rate"] / bo["rate"]
ratio_harness = af["rate"] / ao["rate"]
print()
print("RATIO_MAIN before_orig/after_fast = %.4f   (threshold >= %.1f)  "
      "[ms_per_event %.2f -> %.2f]" % (ratio_main, TARGET, bo["ms"], af["ms"]))
print("RATIO_PSCALIB after_orig/before_orig = %.4f   [evt_per_s %.3f -> %.3f; "
      "the pscalib SOURCE change alone, same unmodified harness]"
      % (ratio_pscalib, bo["rate"], ao["rate"]))
print("RATIO_HARNESS after_fast/after_orig = %.4f   [evt_per_s %.3f -> %.3f; "
      "the harness patch alone, same pscalib]"
      % (ratio_harness, ao["rate"], af["rate"]))
print("RATIO_PRODUCT_CHECK pscalib x harness = %.4f  vs RATIO_MAIN %.4f  "
      "(must agree to rounding)"
      % (ratio_pscalib * ratio_harness, ratio_main))

# ---- the OPTIONAL fifth arm ---------------------------------------------
af2 = R.get("after_fast2", {}).get(W)
if af2:
    print()
    print("RATIO_MAIN2 before_orig/after_fast2 = %.4f   (threshold >= %.1f)  "
          "[ms_per_event %.2f -> %.2f]"
          % (bo["ms"] / af2["ms"], TARGET, bo["ms"], af2["ms"]))
    print("RATIO_BATCH after_fast2/after_fast = %.4f   [evt_per_s %.3f -> %.3f; "
          "the batched float64 accumulate alone, same pscalib and same level-1 patch]"
          % (af2["rate"] / af["rate"], af["rate"], af2["rate"]))
    # The CONSERVATIVE numerator: before_orig reads coldest of all the arms,
    # so quoting it flatters every later arm. before_again is the same code
    # with a fully warm cache, and min() of the two cannot overstate.
    cons = min(bo["ms"], ba["ms"])
    print("RATIO_CONSERVATIVE min(before_orig,before_again)/after_fast  = %.4f"
          % (cons / af["ms"]))
    print("RATIO_CONSERVATIVE min(before_orig,before_again)/after_fast2 = %.4f"
          % (cons / af2["ms"]))
else:
    cons = min(bo["ms"], ba["ms"])
    print("RATIO_CONSERVATIVE min(before_orig,before_again)/after_fast  = %.4f"
          % (cons / af["ms"]))
    print("FAST2_ABSENT the optional fifth arm produced no RESULT lines")

# ---- the cache control ---------------------------------------------------
cache = ba["rate"] / bo["rate"]
cache_ok = 0.90 <= cache <= 1.10
print("CACHE_CONTROL before_again/before_orig = %.4f   [evt_per_s %.3f -> "
      "%.3f]  in_band(0.90..1.10)=%s" % (cache, bo["rate"], ba["rate"], cache_ok))
if not cache_ok:
    print("CACHE_CONTROL_FLAG *** the identical incumbent arm, re-run LAST with "
          "a fully warm page cache, did NOT reproduce its own first number "
          "(%.4fx). Every cache bias in this job points toward PASSING, so "
          "RATIO_MAIN above is inflated by roughly this factor and must be "
          "re-taken (drop caches, or interleave the arms) before it is "
          "published. Flagged, not auto-voided: the printed ratio is the "
          "evidence needed to make that call." % cache)

# ---- symmetry on BYTES READ, never on bandwidth --------------------------
# Every arm reads the SAME 512 events, so the rchar totals must agree. This is
# independent of speed. (The predecessor's MB_per_s rule was the opposite: with
# nbytes fixed, MB_per_s is 1/wall, so it rejected every real speedup. Its rc=96
# carries zero information -- see the header.)
#
# STAGE-4b EDIT: this check is now REPORT-ONLY. It used to set fail = 97 above
# a 1.50 spread; that exit is GONE. Reasons: (a) the campaign brief forbids
# inheriting ANY arm-symmetry reject rule and this was the last one left in the
# file -- the header's item 6 claimed none survived, which was not true while
# `fail = 97` was here; (b) `gb` is reconstructed as an INTEGER MB_per_s times
# wall, and nbytes is the workers' own rchar, i.e. every byte the PROCESS read,
# imports and .so's included, with the level-1 patch's prefetch thread
# contributing from a SECOND thread -- so a pure read-path accounting
# difference could void a run whose arms all exited 0, which is exactly how the
# predecessor's rule destroyed three clean jobs. The spread is PRINTED and
# TAGGED; deciding a large one is disqualifying is a human call, and the
# printed line is the evidence needed to make it. (The 8-event smoke 34323653
# measured max/min=1.012 across all four arms, so the accounting does in fact
# agree across the prefetch-threaded and unthreaded read paths.)
for w in (1, 16):
    vals = {a: R[a][w]["gb"] for a in ARMS if w in R[a]}
    if len(vals) < 2:
        continue
    lo, hi = min(vals.values()), max(vals.values())
    spread = (hi / lo) if lo > 0 else float("inf")
    tag = "OK"
    if spread > 1.50:
        tag = "LARGE(report-only, does NOT void the job)"
    elif spread > 1.15:
        tag = "WARN"
    print("BYTES_SYMMETRY workers=%d max/min=%.3f %s  %s"
          % (w, spread, tag,
             " ".join("%s=%.2fGB" % (a, vals[a]) for a in ARMS if a in vals)))

# ---- workers=16, printed for the record, NOT thresholded -----------------
if all(16 in R[a] for a in ARMS):
    b16, a16, f16, g16 = (R[a][16] for a in ARMS)
    print()
    print("W16_RECORD RATIO_MAIN before_orig/after_fast = %.4f  "
          "[ms_per_event %.2f -> %.2f]  (FOR THE RECORD ONLY -- the campaign "
          "target is defined at workers=1 and this line is NOT thresholded)"
          % (b16["ms"] / f16["ms"], b16["ms"], f16["ms"]))
    print("W16_RECORD RATIO_PSCALIB after_orig/before_orig = %.4f"
          % (a16["rate"] / b16["rate"]))
    print("W16_RECORD RATIO_HARNESS after_fast/after_orig = %.4f"
          % (f16["rate"] / a16["rate"]))
    print("W16_RECORD CACHE_CONTROL before_again/before_orig = %.4f"
          % (g16["rate"] / b16["rate"]))
    for a in ARMS:
        print("W16_RECORD scaling arm=%s 1->16 workers = %.2fx  (%.2f -> %.2f "
              "evt/s)" % (a, R[a][1]["wall"] / R[a][16]["wall"], R[a][1]["rate"],
                          R[a][16]["rate"]))

# ---- the harness's own per-arm verdicts ----------------------------------
bad_arm = [(a, w) for a in ARMS for w in sorted(R[a])
           if R[a][w]["verdict"] != "PASS" or R[a][w]["pinv"] != 1]
if bad_arm:
    print("PROOF_VERDICT harness self-verdict FAILED for %s" % (bad_arm,))
    fail = fail or 98

target_met = ratio_main >= TARGET
print()
print("PROOF_VERDICT ratio_main=%.4f target=%.1f target_met=%s "
      "ratio_pscalib=%.4f ratio_harness=%.4f cache_control=%.4f "
      "cache_control_ok=%s arms_ok=%s"
      % (ratio_main, TARGET, target_met, ratio_pscalib, ratio_harness, cache,
         cache_ok, not bad_arm))
if fail:
    sys.exit(fail)
# 90 == "measured cleanly, but BELOW the target". Distinct from every integrity
# exit code above, so a below-target measurement is never mistaken for a broken
# job (and vice versa).
sys.exit(0 if target_met else 90)
PYVERDICT
vrc=$?
echo "PROOF_VERDICT_EXIT rc=$vrc"

# ==========================================================================
# STEP 5 -- UN-CLOCKED IN-SITU PHASE PROFILE.  Runs AFTER every number above is
# already printed, so it cannot perturb one of them, and it is wrapped so that
# a failure here can NEVER change the job's exit code.
#
# WHY.  The campaign's accumulate cost is QUARANTINED: 43.94 ms/event was
# measured with ONE bin while the harness runs --bins 10, and in-situ smokes put
# it at 49-72 ms.  This step imports each harness and profiles its OWN
# accumulate_slice() over 64 real events at nbins=10, then over the SAME 64
# events with every bin forced to 0 -- so the bin-count effect is measured, not
# argued about -- then over 256 and 512 events.  512 is the ARM's own event
# count, so that line needs no extrapolation at all; 64 alone would not do,
# because with subbatch=16 it is only 4 batches (prefetch overlap ceiling
# 1-1/4 = 0.75) and 10 of its 64 events are the first in their bin and pay the
# 134 MB astype allocation (16%, against 2% in the arm).  A 4-event UNTIMED
# warmup runs first so pscalib's one-time ~1.5 s gain/mask memo derivation is
# not amortised over the probe (it would add ~23 ms/event to calib_ms at n=64).
# Both harnesses are profiled with the CANDIDATE pscalib, so the difference
# between the two INSITU lines IS the harness patch's contribution.
# ==========================================================================
echo "########## STEP5_INSITU job=$SLURM_JOB_ID $(date) ##########"
export PYTHONPATH=$PSC_AFTER/src:$PSD/src:$BASE_PP
export PYTHONDONTWRITEBYTECODE=1
unset BENCH_FAST_BATCH
unset BENCH_WORKERS
echo "STEP5 pscalib=$PSC_AFTER sha=$(git -C $PSC_AFTER rev-parse HEAD) dirty=$(git -C $PSC_AFTER status --porcelain | wc -l) src_md5=$(srcmd5 $PSC_AFTER)"
echo "STEP5 npz=$NPZ"
STEP5_PAIRS="fast:$HARNESS_FAST orig:$HARNESS_ORIG"
# STAGE-4b EDIT (9): profile the LEVEL-2 harness too when its arm ran. Its
# accumulate is the term the next iteration has to attack, so a decomposition
# of it is worth ~2 minutes of an un-clocked step. BENCH_FAST_BATCH is read at
# IMPORT time by bench_cube_calib_fast2.py, so it is exported for exactly that
# invocation and unset again immediately after.
[ "$RUN_FAST2" = "1" ] && STEP5_PAIRS="$STEP5_PAIRS fast2:$HARNESS_FAST2"
for pair in $STEP5_PAIRS; do
  tag=${pair%%:*}; hf=${pair#*:}
  if [ "$tag" = "fast2" ]; then export BENCH_FAST_BATCH=$FAST2_BATCH;
  else unset BENCH_FAST_BATCH; fi
  echo "---------- INSITU harness=$tag file=$hf md5=$(filemd5 $hf) BENCH_FAST_BATCH=${BENCH_FAST_BATCH:-<unset>} ----------"
  python $INSITU --harness "$hf" --label "$tag" --n 64 --nbins $BINS \
                 --pool $EVENTS --bin-var $BINVAR --npz $NPZ --warmup 4 \
                 --also-one-bin --extra-n 256,$EVENTS \
                 2>&1 | tee $LOGDIR/insitu_$tag.log
  echo "INSITU_EXIT tag=$tag rc=${PIPESTATUS[0]} (ADVISORY -- cannot change the job's exit code)"
  unset BENCH_FAST_BATCH
done
echo "########## STEP5_INSITU_DONE $(date) ##########"
echo "--- SQUEUE_AT_END ---"
squeue -u "$(whoami)" -o "%.10i %.14j %.9T %.14R %.6D %.5C %.11M %.11l" 2>&1
echo "--- end SQUEUE_AT_END ---"
echo "NODE_AT_END $(hostname)  DATE=$(date -Is)"

if [ $gate_bad -ne 0 ]; then
  echo 'FATAL: BITGATE/OUTGATE did not prove byte-exactness in every arm/mode -- every ratio above is meaningless'
  echo "cube-proof-4arm done rc=95 $(date)"
  exit 95
fi
if [ $rc_all -ne 0 ]; then
  echo "FATAL: an arm exited non-zero (rc=$rc_all)"
  echo "cube-proof-4arm done rc=$rc_all $(date)"
  exit $rc_all
fi
echo "cube-proof-4arm done rc=$vrc $(date)"
exit $vrc
