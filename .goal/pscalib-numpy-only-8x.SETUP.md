# Setup — pscalib-numpy-only-8x

Read this **before** issuing any remote command. It is the portable half of the
contract: everything that depends on the machine you are running from, rather
than on the remote, lives here.

The remote host is invariant: **`sdfiana025`** at SLAC S3DF. Nothing in this
campaign runs computation locally.

---

## What must travel to a new machine

Four files, all in `.goal/`:

```
.goal/pscalib-numpy-only-8x.json          the contract
.goal/pscalib-numpy-only-8x.ledger.json   the ledger (may hold a run in progress)
.goal/pscalib-numpy-only-8x.claims.json   the claims file
.goal/pscalib-numpy-only-8x.SETUP.md      this file
```

That set is self-contained. The predecessor campaign's record
(`pscalib-calib-speedup.{json,ledger.json,claims.json,SETUP.md}`) is useful
background — it is where every measured number in the contract's `baseline`,
`instrument` and `gate_design` blocks came from — but it is **not a
dependency**. This campaign runs without it present.

Nothing in these four files names a path on the machine you run from. Every
absolute path is on the remote, and that is deliberate: the local side must
stay portable, the remote side is pinned.

**Run `/goal` from the directory that contains `.goal/`.** The contract's ledger
and claims paths are relative, so a different cwd writes the files somewhere
unintended and the printed digest stops matching the file on disk. The name of
that containing directory does not matter.

---

## Human preconditions (an agent cannot satisfy these)

Both must be true before iteration 1. Neither is something a subagent can fix,
because one needs a package install and the other needs an interactive MFA
prompt.

### 1. The cc-bridge CLI is installed locally

```bash
command -v bridge bridge-session
```

Expect two paths. If either is missing, **stop and tell the user** — there is no
fallback, and no amount of retrying will conjure the CLI.

The bridge skill is called `/bridge-skills`.

### 2. SSH to `sdfiana025` works non-interactively

```bash
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=15 sdfiana025 hostname
```

Expect `sdfiana025`. `BatchMode=yes` is deliberate: it fails fast rather than
hanging on a password or MFA prompt.

If it fails, the cause is almost always one of:

| Symptom | Cause | Who fixes it |
|---|---|---|
| `Could not resolve hostname` | no `~/.ssh/config` entry | user (see below) |
| `Permission denied (publickey)` | no key, or key not registered at SLAC | user |
| Hangs, or `Host key verification failed` | MFA session not established | user, interactively |

**MFA is the common case on a fresh machine.** S3DF requires it through the
jump host, and it cannot be automated. The correct agent behaviour is to report
the blocker plainly and ask the user to run the login themselves — suggest they
type `! ssh s3dflogin-mfa.slac.stanford.edu` in the prompt so the output lands
in the conversation. Do **not** spend iterations working around it, and do not
mark any done_when item met on a machine that cannot reach the remote.

### The SSH config this campaign assumes

If `~/.ssh/config` lacks it, the user needs:

```
Host s3dflogin-mfa.slac.stanford.edu
  HostName s3dflogin-mfa.slac.stanford.edu
  User cwang31

Host sdfiana*
  User cwang31
  ProxyJump s3dflogin-mfa.slac.stanford.edu

Host sdfmilan*
  User cwang31
  ProxyJump s3dflogin-mfa.slac.stanford.edu
```

A `ControlMaster` socket under `~/.ssh/sockets` makes repeat connections fast
and avoids re-prompting for MFA. Its presence is an optimisation, not a
requirement — but if it exists and is live, connections are near-instant, which
is a useful signal that auth is already established.

---

## Bridge sessions: fresh for this campaign, at most three

**Start your own sessions and do not inherit any.** A session left over from a
human's interactive work — or from the predecessor campaign — may have a
different `--root-dir`, may be stale, or may be in use, and sessions serialize,
so sharing one silently blocks your commands behind someone else's.

**At most 3 concurrent sessions.** One session runs one command at a time, so
three is the parallelism ceiling for remote work: a build in one, a source
sweep in another, a poll of a running job in the third. Three parallel sessions
was the difference between 40 minutes and 2 hours in the predecessor campaign's
search work — but a fourth buys nothing and makes the failure modes harder to
read.

```bash
bridge-session start --name pscalib2-bench -- \
  ssh sdfiana025 'uv run ~/bridge-server --root-dir /sdf'
bridge --session pscalib2-bench status
```

Naming convention: `pscalib2-<purpose>` — `pscalib2-bench`, `pscalib2-src`,
`pscalib2-verify`. The `2` prefix matters: a bare `pscalib-*` name collides with
the predecessor campaign's sessions, and `s1`/`s2` are what a human at a
terminal uses, so colliding with one is how you end up debugging someone else's
session.

**Restart rather than reuse.** At the start of the campaign, list what is
running:

```bash
bridge-session list
```

If a leftover `pscalib-*` session from the predecessor campaign is holding a
slot you need, stop it — it is campaign infrastructure, not a person's work:

```bash
bridge-session stop pscalib-src
```

Never stop a session with a human-style name (`s1`, `s2`, …). Stop all of your
own when the iteration ends.

Use `--root-dir /sdf` (wide). A narrow root only restricts `bridge read` paths;
absolute paths inside `bridge bash` work regardless, so a narrow root buys
nothing and costs you reads.

---

## Bridge command essentials

| Command | Purpose |
|---|---|
| `bridge --session <n> status` | confirm alive |
| `bridge --session <n> bash "<cmd>"` | run remotely; prefer `rg` / `fd` |
| `bridge --session <n> read <path> --raw > local` | download without filling context |
| `bridge --session <n> write <path> --file <local>` | upload |

Four gotchas that cost real time:

1. **`--timeout` goes before the subcommand.** `bridge --timeout 600 bash "..."`,
   never `bridge bash --timeout 600`.
2. **Two independent timeouts.** The remote one above, and the local Bash tool's
   own 120 s. For anything long, redirect to a remote file and poll it:
   `bridge ... bash "cmd > /tmp/out.txt 2>&1 &"` then read `/tmp/out.txt`.
3. **Output caps at 1 MB.** Scope `rg`/`fd` or redirect remotely and filter
   remotely — keeping the list on the remote side lets you re-filter for free.
4. **A command that returns too fast with no output is broken, not a negative
   result.** Quoting errors fail silently. Re-check before believing an absence.

---

## Remote invariants

Confirmed on `sdfiana025` on **2026-08-06**, first-hand, while authoring this
campaign:

| Tool | Path / version |
|---|---|
| `bridge-server` | `/sdf/home/c/cwang31/bridge-server`, starts under `uv run` |
| `uv` | on PATH (needed for `uv run ~/bridge-server`) |
| `gh` | `~/.local/bin/gh` 2.62.0, authenticated as `carbonscott`, `repo` scope |
| `squeue` / `srun` | `/opt/slurm/slurm-curr/bin/` |
| pscalib `origin` | `git@github.com:carbonscott/pscalib.git`, SSH auth working |

Carried over from the predecessor campaign, verified 2026-08-05 but **not**
re-checked since — re-verify rather than trusting the table:

| Tool | Path |
|---|---|
| `rg` | `/usr/bin/rg` |
| `fd` | `~/.local/bin/fd` |
| `gcc` | `/usr/bin/gcc` (13.3.0) |
| `python3` | uv-managed CPython 3.12.12 |

One command re-verifies the lot:

```bash
bridge --session pscalib2-bench bash "command -v rg fd gcc python3 srun uv gh; ls ~/bridge-server"
```

The `/sdf` paths in the contract's `context.paths` are all on shared project
storage and are stable. Two cautions carried over: `/sdf/scratch` is
**purgeable**, so never leave the only copy of anything there; and
`/sdf/data/lcls/ds` contains both lower- and upper-case hutch directories
(`mfx` and `MFX`), which inflates any sweep that does not account for it.

---

## Measurement environment

The baseline this campaign must beat by 8× was measured on **milano** nodes
(`sdfmilan053`, `sdfmilan046`) via slurm, 120 CPU / 503 GiB, with the psana
release at `/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_070626`.

Measure candidates the same way. A number from the `sdfiana025` login node is a
smoke test only — it is shared, and this workload is memory-bandwidth bound, so
login-node timings are both noisy and pessimistic and are **not** comparable to
the baseline.

### The three-arm design, and why

The contract requires **three arms in one job**, not two:

| Arm | pscalib | harness | measures |
|---|---|---|---|
| `before_orig` | incumbent `c5ce538` | unmodified | the ~645 ms/event baseline, re-measured in-job |
| `after_orig` | candidate | unmodified | what pscalib alone bought |
| `after_fast` | candidate | patched | the 8.0× claim |

The threshold ratio is `before_orig / after_fast`, computed inside one job on
one node. This exists because the predecessor campaign anchored its headline on
a July constant and then found two of three incumbent arms running 33% slower
than July for reasons nobody could explain — external load did not account for
it. Re-measuring the denominator in the same job on the same node removes that
failure mode entirely, and yields the harness patch's own contribution
(`after_fast / after_orig`) for free.

Three rules that follow, each learned expensively:

- **512 events, never 128.** At 128 events, bit-identical code produced ratios
  of 4.31× and 1.33× depending on arm order, driven by page-cache warming.
- **Serialize your own jobs and check exclusivity before submitting.** Three
  predecessor runs were destroyed by an I/O storm from that campaign's own
  concurrent xtc readers. `--exclusive` protects the CPUs, not Weka.
- **Check arm symmetry by bytes read, not MB/s.** `MB_per_s` is `nbytes/wall`
  with `nbytes` fixed by the event count, so an MB/s symmetry rule is a
  wall-time symmetry test in disguise, and any real speedup above ~1.18× must
  fail it. The predecessor's three proof jobs all exited `rc=96` for exactly
  this reason while every arm exited `rc=0`. If you keep such a rule, treat
  `rc=96` as carrying no information and say so in the ledger.

---

## Failure modes, and what they look like

| What you see | What it means | What to do |
|---|---|---|
| `bridge status` → no active session | you have not started one | start one; do not hunt for someone else's |
| `bridge-session start` fails, 3 already running | at the session cap | stop one of yours, or a leftover `pscalib-*` |
| `ssh` hangs with no output | MFA needed | stop, report to the user, ask them to log in |
| `bridge bash` returns instantly, empty | quoting error, not an empty result | fix the quoting, re-run |
| `bridge bash` output truncated | hit the 1 MB cap | redirect remotely, filter remotely |
| Command killed at ~120 s | local Bash timeout | background it remotely, poll a file |
| slurm job pending indefinitely | queue contention | record the job id, poll; do not resubmit blind |
| Timings ~3× worse than baseline | measured on the login node | re-run under `srun` on milano |
| `ValueError` before argparse | `BENCH_WORKERS` is set and split on commas | unset it in the sbatch |
| `before_orig` arm far off 645 ms | the instrument is not reproducing the baseline | do not quote the ratio; diagnose first |

---

## What is not verified

The 2026-08-06 rows in the invariants table were checked first-hand from one
machine; the 2026-08-05 rows were not re-checked. The remote is shared and
mutable — re-run the verification command rather than trusting either set. The
SSH config block reflects what worked on the authoring machine; a different
machine may need `IdentityFile` or a different jump-host alias. Nothing here has
been tested from a second machine — this document is the prediction, and the
first run elsewhere is its test.
