# Setup — pscalib-calib-speedup

Read this **before** issuing any remote command. It is the portable half of the
contract: everything that depends on the machine you are running from, rather
than on the remote, lives here.

The remote host is invariant: **`sdfiana025`** at SLAC S3DF. Nothing in this
campaign runs computation locally.

---

## What must travel to a new machine

Four files, all in `.goal/`:

```
.goal/pscalib-calib-speedup.json          the contract
.goal/pscalib-calib-speedup.ledger.json   the ledger (may hold a run in progress)
.goal/pscalib-calib-speedup.claims.json   the claims file
.goal/pscalib-calib-speedup.SETUP.md      this file
```

That set is self-contained. The campaign's `.ai/research/*.md` notes are useful
background but **not required** — every fact the loop needs is in the contract's
`context` block.

**Run `/goal` from the directory that contains `.goal/`.** The contract's ledger
and claims paths are relative, so a different cwd writes the files somewhere
unintended and the printed digest stops matching the file on disk.

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

Bridge skill is called `/bridge-skills`.

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

## Bridge sessions: always fresh, never inherited

**Do not reuse a session you did not create, and do not assume one exists.** A
session left over from a human's interactive work may have a different
`--root-dir`, may be stale, or may be in use — and sessions serialize, so
sharing one silently blocks your own commands behind someone else's.

Start your own, named for its purpose so concurrent streams do not collide:

```bash
bridge-session start --name pscalib-bench -- \
  ssh sdfiana025 'uv run ~/bridge-server --root-dir /sdf'
bridge --session pscalib-bench status
```

Naming convention: `pscalib-<purpose>` — e.g. `pscalib-bench`, `pscalib-src`,
`pscalib-verify`. Never plain `s1`, `s2` — those are what a human at a terminal
uses, and colliding with one is how you end up debugging someone else's session.

**One session runs one command at a time.** For concurrent work (say, a build
in one place and a source sweep in another), start one session per stream. Three
parallel sessions was the difference between 40 minutes and 2 hours in this
campaign's earlier search work.

Use `--root-dir /sdf` (wide). A narrow root only restricts `bridge read` paths;
absolute paths inside `bridge bash` work regardless, so a narrow root buys
nothing and costs you reads.

Stop your sessions when the iteration is done:

```bash
bridge-session list
bridge-session stop pscalib-bench
```

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

## Remote invariants (verified 2026-08-05)

Confirmed present on `sdfiana025`:

| Tool | Path |
|---|---|
| `rg` | `/usr/bin/rg` |
| `fd` | `~/.local/bin/fd` |
| `gcc` | `/usr/bin/gcc` (13.3.0) |
| `python3` | uv-managed CPython 3.12.12 |
| `srun` | `/opt/slurm/slurm-curr/bin/srun` |
| `bridge-server` | `/sdf/home/c/cwang31/bridge-server` |
| `uv` | on PATH (needed for `uv run ~/bridge-server`) |

Re-verify in one command at the start of a campaign:

```bash
bridge --session pscalib-bench bash "command -v rg fd gcc python3 srun uv; ls ~/bridge-server"
```

The `/sdf` paths the contract's `context.paths` names are all on shared project
storage and are stable. Two cautions carried over from earlier work: `/sdf/scratch`
is **purgeable**, so never leave the only copy of anything there; and
`/sdf/data/lcls/ds` contains both lower- and upper-case hutch directories
(`mfx` and `MFX`), which inflates any sweep that does not account for it.

---

## Measurement environment

The baseline runs this campaign must beat were measured on **milano** nodes
(`sdfmilan053`, `sdfmilan046`) via slurm, 120 CPU / 503 GiB, with the psana
release at `/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_070626`.

Measure candidates the same way. A number from the `sdfiana025` login node is a
smoke test only — it is shared, and this workload is memory-bandwidth bound, so
login-node timings are both noisy and pessimistic and are **not** comparable to
the baseline.

The contract requires the paired `before`/`after` arms in a **single** job for
exactly this reason: it self-normalizes against node-to-node variation, so you
never have to argue that two separate jobs landed on equivalent hardware.

---

## Failure modes, and what they look like

| What you see | What it means | What to do |
|---|---|---|
| `bridge status` → no active session | you have not started one | start one; do not hunt for someone else's |
| `ssh` hangs with no output | MFA needed | stop, report to the user, ask them to log in |
| `bridge bash` returns instantly, empty | quoting error, not an empty result | fix the quoting, re-run |
| `bridge bash` output truncated | hit the 1 MB cap | redirect remotely, filter remotely |
| Command killed at ~120 s | local Bash timeout | background it remotely, poll a file |
| slurm job pending indefinitely | queue contention | record the job id, poll; do not resubmit blind |
| Timings ~3× worse than baseline | measured on the login node | re-run under `srun` on milano |

---

## What is not verified

The tool inventory and paths above were checked on 2026-08-05 from one machine.
The remote is shared and mutable: re-run the verification command rather than
trusting this table. The SSH config block reflects what worked on the authoring
machine; a different machine may need `IdentityFile` or a different jump-host
alias. Nothing here has been tested from a second machine — this document is the
prediction, and the first run elsewhere is its test.
