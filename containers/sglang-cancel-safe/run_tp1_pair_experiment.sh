#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 PATCHED_IMAGE.sif MODEL_PATH [PORT_A] [PORT_B]" >&2
    exit 2
fi
[[ -n ${SLURM_JOB_ID:-} ]] || {
    echo "run inside a dedicated two-GPU Slurm allocation" >&2
    exit 2
}

image=$(readlink -e -- "$1")
model_path=$(readlink -e -- "$2")
port_a=${3:-18792}
port_b=${4:-18793}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
control_dir=${SCITEX_EXPERIMENT_CONTROL_DIR:-}

if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
elif [[ -x /apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Apptainer/1.3.3/bin/apptainer ]]; then
    runtime=/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Apptainer/1.3.3/bin/apptainer
else
    echo "apptainer is required" >&2
    exit 1
fi

mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
[[ ${#gpu_memory[@]} -eq 2 ]] || {
    echo "experiment requires exactly two visible GPUs" >&2
    exit 2
}
for used in "${gpu_memory[@]}"; do
    (( used < 2048 )) || {
        echo "GPU memory is in use (${used} MiB); refusing to disturb a live server" >&2
        exit 2
    }
done

"${runtime}" test "${image}"
if [[ -n ${control_dir} ]]; then
    [[ -d ${control_dir} ]] || {
        echo "control directory does not exist: ${control_dir}" >&2
        exit 2
    }
    run_dir=$(mktemp -d -- "${control_dir}/tp1-run.XXXXXX")
    printf '%s\n' "$$" >"${control_dir}/tp1-supervisor.pid"
    printf '%s\n' "${run_dir}" >"${control_dir}/tp1-run-dir"
else
    run_dir=$(mktemp -d -- /data/scratch/sglang-tp1-pair.XXXXXX)
fi
for name in qwen38-27b-a qwen38-27b-b; do
    mkdir -p -- "${run_dir}/${name}-cache"
    "${runtime}" exec --bind "${run_dir}" "${image}" python3 -c \
        'from pathlib import Path; p=Path(__import__("sys").argv[1]); p.write_text("writable\n"); assert p.read_text() == "writable\n"' \
        "${run_dir}/${name}-cache/container-write-preflight"
done
pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
        fi
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    echo "experiment artifacts retained at ${run_dir}" >&2
}
trap cleanup EXIT

json_override='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'

launch_replica() {
    local gpu=$1 port=$2 name=$3
    mkdir -p -- "${run_dir}/${name}-cache"
    CUDA_VISIBLE_DEVICES=${gpu} \
    SGLANG_CACHE_DIR="${run_dir}/${name}-cache/sglang" \
    SGLANG_JIT_CACHE_DIR="${run_dir}/${name}-cache/jit" \
    DG_JIT_CACHE_DIR="${run_dir}/${name}-cache/deepgemm" \
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    "${runtime}" exec --nv \
        --bind "${model_path}" \
        --bind "${run_dir}" \
        "${image}" \
        python3 -m sglang.launch_server \
        --model-path "${model_path}" \
        --served-model-name "${name}" \
        --tp-size 1 \
        --mem-fraction-static 0.75 \
        --context-length 400000 \
        --max-running-requests 4 \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend flashinfer \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        --disable-prefill-cuda-graph \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --json-model-override-args "${json_override}" \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --enable-metrics \
        --host 127.0.0.1 \
        --port "${port}" \
        >"${run_dir}/${name}.log" 2>&1 &
    pids+=("$!")
}

launch_replica 0 "${port_a}" qwen38-27b-a
launch_replica 1 "${port_b}" qwen38-27b-b

for port in "${port_a}" "${port_b}"; do
    deadline=$((SECONDS + 900))
    until curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; do
        for pid in "${pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null || {
                echo "a TP=1 server exited before health" >&2
                exit 1
            }
        done
        (( SECONDS < deadline )) || {
            echo "server on ${port} did not become healthy" >&2
            exit 1
        }
        sleep 2
    done
done

nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used \
    --format=csv,noheader >"${run_dir}/gpu-allocation.csv"
curl --fail --silent "http://127.0.0.1:${port_a}/v1/loads?include=core" \
    >"${run_dir}/loads-a-before.json"
curl --fail --silent "http://127.0.0.1:${port_b}/v1/loads?include=core" \
    >"${run_dir}/loads-b-before.json"

python3 "${script_dir}/capacity_probe.py" \
    --base-url "http://127.0.0.1:${port_a}" \
    --output "${run_dir}/capacity-a.json"
python3 "${script_dir}/capacity_probe.py" \
    --base-url "http://127.0.0.1:${port_b}" \
    --output "${run_dir}/capacity-b.json"
python3 "${script_dir}/acceptance_disconnect.py" \
    --base-url "http://127.0.0.1:${port_a}" \
    --model qwen38-27b-a
python3 "${script_dir}/acceptance_disconnect.py" \
    --base-url "http://127.0.0.1:${port_b}" \
    --model qwen38-27b-b

echo "PASS: TP=1 pair capacity and disconnect results are in ${run_dir}"
touch "${run_dir}/READY"
echo "holding both healthy TP=1 replicas; terminate this step to stop them"
wait
