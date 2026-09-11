#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 /scratch/PATH/IMAGE.sif" >&2
    exit 2
}

[[ $# -eq 1 ]] || usage
requested_output=$1
[[ ${requested_output} = /* ]] || usage

output=$(readlink -m -- "${requested_output}")
case "${output}" in
    /scratch/*.sif|/data/scratch/*.sif) ;;
    *)
        echo "refusing output outside scratch storage or without .sif suffix: ${output}" >&2
        exit 2
        ;;
esac
[[ ! -e ${output} ]] || {
    echo "refusing to overwrite existing image: ${output}" >&2
    exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=$(dirname -- "${output}")
mkdir -p -- "${output_dir}"
build_tmp=$(mktemp -d -- "${output_dir}/.sglang-cancel-safe.XXXXXX")
cleanup() {
    rm -rf -- "${build_tmp}"
}
trap cleanup EXIT

if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    runtime=singularity
else
    echo "apptainer or singularity is required" >&2
    exit 1
fi

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${build_tmp}/cache}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}
mkdir -p -- "${APPTAINER_CACHEDIR}"

(
    cd -- "${script_dir}"
    "${runtime}" build "${build_tmp}/image.sif" Apptainer.def
)
"${runtime}" test "${build_tmp}/image.sif"
"${runtime}" inspect --labels "${build_tmp}/image.sif" \
    > "${build_tmp}/labels.json"
grep -q '4ccff141dbe992794f9da6c3aa23535b4f72000d' \
    "${build_tmp}/labels.json"

mv -- "${build_tmp}/image.sif" "${output}"
echo "built ${output}"
