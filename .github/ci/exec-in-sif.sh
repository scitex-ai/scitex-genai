#!/usr/bin/env bash
# Outer apptainer-exec wrapper for scitex-tex's self-hosted (Spartan) CI.
#
# Runs ON THE RUNNER (outside the SIF). Resolves the apptainer shim + SIF image
# from the repo Actions Variables, then `apptainer exec`s the SIF and hands off
# to an INNER script (run inside the container). Keeps every workflow job's YAML
# down to one line — `bash .github/ci/exec-in-sif.sh <inner-script> [args...]` —
# and concentrates all the HPC/SIF plumbing (shim PATH, ~-expansion, scratch,
# binds) in one version-controlled place.
#
# Required env (set by the workflow from repo Actions Variables):
#   SCITEX_CI_APPTAINER   path to the apptainer shim   (e.g. ~/.env-3.11/bin/apptainer)
#   SCITEX_CI_SIF         path to the CI SIF image     (e.g. ~/.scitex/dev/containers/ci-cpu.sif)
#
# Usage:
#   bash .github/ci/exec-in-sif.sh run-in-sif.sh 3.12
#
# Fail-loud (operator directive): a missing shim or SIF is a HARD error — never
# a silent fallback to a bare-runner install.
set -euo pipefail

INNER="${1:?inner script name required (relative to .github/ci/)}"
shift || true

# The runner's job shell is --noprofile --norc (no Lmod), so the apptainer shim
# must be put on PATH explicitly; it execs the real Apptainer binary directly.
# ~-expand the Actions-Variable paths: a quoted "~/…" is NOT tilde-expanded by
# the shell, so substitute a leading ~ with $HOME ourselves.
APPTAINER="${SCITEX_CI_APPTAINER:?SCITEX_CI_APPTAINER not set (repo Actions Variable)}"
SIF="${SCITEX_CI_SIF:?SCITEX_CI_SIF not set (repo Actions Variable)}"
APPTAINER="${APPTAINER/#\~/$HOME}"
SIF="${SIF/#\~/$HOME}"
export PATH="$HOME/.env-3.11/bin:$PATH"

# The declared path is a per-repo Actions Variable and it is often wrong for the
# runner the job actually landed on: measured 2026-09-06, 64 of 77 repos declare
# ~/.env-3.11/bin/apptainer, which exists on Spartan and NOT on the scitex-org-cpu
# runners, where apptainer is /usr/bin/apptainer. Prefer the declaration, fall back
# to whatever is on PATH, and fail loudly naming BOTH when neither resolves — a
# variable that names an absent path must not read the same as no apptainer at all.
if [ ! -x "$APPTAINER" ]; then
    FALLBACK="$(command -v apptainer 2>/dev/null || true)"
    if [ -n "$FALLBACK" ]; then
        echo "::warning::declared apptainer '$APPTAINER' is not executable here; using '$FALLBACK' from PATH"
        APPTAINER="$FALLBACK"
    fi
fi
[ -x "$APPTAINER" ] || {
    echo "::error::no usable apptainer: SCITEX_CI_APPTAINER='${SCITEX_CI_APPTAINER}' is not executable and none is on PATH"
    exit 1
}
[ -f "$SIF" ] || {
    echo "::error::CI SIF missing at $SIF — rebuild it: scitex-container apptainer build ci-cpu"
    exit 1
}

# Apptainer scratch, and the project bind, are SPARTAN facts — used where Spartan
# is, skipped where it is not. On Spartan the shared project filesystem keeps HOME
# clean and $HOME/.scitex is a symlink into punim0264, so the bind is what makes
# that symlink resolve inside the container. On the scitex-org-cpu runners neither
# is true: /data does not exist, `mkdir -p` on it fails with EACCES, and the whole
# job dies there before a single test runs (measured 2026-09-06 on run 34009594049:
# "mkdir: cannot create directory '/data': Permission denied"). Deciding by what is
# PRESENT rather than by which pool we assume we are on is what makes this wrapper
# run wherever the job was scheduled.
SPARTAN_PROJECT="/data/gpfs/projects/punim0264"
BIND_ARGS=()
if [ -d "$SPARTAN_PROJECT" ]; then
    export APPTAINER_TMPDIR="$SPARTAN_PROJECT/ywatanabe/ci/apptainer-tmp"
    BIND_ARGS=(--bind "$SPARTAN_PROJECT")
else
    export APPTAINER_TMPDIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/apptainer-tmp"
fi
mkdir -p "$APPTAINER_TMPDIR" || {
    echo "::error::cannot create apptainer scratch at $APPTAINER_TMPDIR"
    exit 1
}

# --pwd "$PWD" keeps the checkout as cwd.
exec "$APPTAINER" exec --pwd "$PWD" "${BIND_ARGS[@]}" \
    "$SIF" bash ".github/ci/$INNER" "$@"
