#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run this fixture inside an existing isolated SLURM step}"
: "${SLURM_STEP_ID:?run this fixture inside an existing isolated SLURM step}"
: "${SCITEX_GENAI_CANARY_STORE_SSH:?set the SSH destination that can reach scitex-primary}"
: "${SCITEX_GENAI_CANARY_STORE_PROXY_COMMAND:?set the validated SSH ProxyCommand}"

PYTHON=${SCITEX_GENAI_SERVE_PYTHON:-python3}
SOURCE_ROOT=${SCITEX_GENAI_CANARY_SOURCE_ROOT:?set the staged committed source root}
STORE_PORT=${SCITEX_GENAI_CANARY_STORE_PORT:-35432}
STORE_ROLE=${SCITEX_GENAI_CANARY_STORE_ROLE:-${USER}__scitex-genai}
CONTROL_SOCKET=/tmp/scitex-genai-canary-store-${SLURM_JOB_ID}-${SLURM_STEP_ID}.sock

cleanup() {
  ssh -S "$CONTROL_SOCKET" -O exit "$SCITEX_GENAI_CANARY_STORE_SSH" \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# HPC compute nodes cannot resolve the fleet overlay name ``scitex-primary``.
# This experiment-only fixture carries the canonical 55432 connection through
# the same authenticated compute-04 SSH path as serving tunnels. There is no
# alternate store and no file fallback: ExitOnForwardFailure and the canary
# publisher both fail closed before the GPU engine starts.
ssh -M -S "$CONTROL_SOCKET" -fNT \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -o "ProxyCommand=$SCITEX_GENAI_CANARY_STORE_PROXY_COMMAND" \
  -L "$STORE_PORT:scitex-primary:55432" \
  "$SCITEX_GENAI_CANARY_STORE_SSH"

export SCITEX_STORE_DSN="postgresql://${STORE_ROLE}@127.0.0.1:${STORE_PORT}/scitex"
export SCITEX_GENAI_CANARY_PURPOSE=qwen38-tp1-context-concurrency
export PYTHONPATH="$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m scitex_genai.serve._cli \
  qwen38-tp1-context-concurrency \
  --models-dir "$SOURCE_ROOT/examples/serve/canary"
