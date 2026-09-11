#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 IMAGE.sif MODEL_PATH [PORT]" >&2
    exit 2
fi
[[ -n ${SLURM_JOB_ID:-} ]] || {
    echo "run inside a dedicated Slurm GPU allocation, not on a login node" >&2
    exit 2
}

image=$(readlink -e -- "$1")
model_path=$2
port=${3:-18792}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=$(mktemp -d -- "/scratch/sglang-cancel-acceptance.XXXXXX")
log_file=${run_dir}/server.log
server_pid=
cleanup() {
    if [[ -n ${server_pid} ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}"
        wait "${server_pid}" || true
    fi
    echo "acceptance artifacts retained at ${run_dir}" >&2
}
trap cleanup EXIT

apptainer exec --nv "${image}" python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tp-size 2 \
    --context-length 1048576 \
    --max-running-requests 8 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --speculative-algorithm EAGLE \
    --enable-metrics \
    >"${log_file}" 2>&1 &
server_pid=$!

deadline=$((SECONDS + 900))
until curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; do
    kill -0 "${server_pid}" 2>/dev/null || {
        tail -100 "${log_file}" >&2
        exit 1
    }
    (( SECONDS < deadline )) || {
        echo "server did not become healthy" >&2
        exit 1
    }
    sleep 2
done

python3 "${script_dir}/acceptance_disconnect.py" \
    --base-url "http://127.0.0.1:${port}" \
    --model "${model_path}" \
    --log "${log_file}"

