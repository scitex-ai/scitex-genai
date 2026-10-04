#!/usr/bin/env bash
# Runs INSIDE the reused scitex-ci SIF (apptainer exec — invoked via
# exec-in-sif.sh, like run/build/publish). $1 = python version, $2 = release
# version without leading v.
#
# Qualifies downloaded release artifacts BEFORE any OIDC token mint:
# every dist/* wheel/sdist must carry Name scitex-genai + matching Version
# from its own METADATA/PKG-INFO (not the filename). Uses the explicit baked
# interpreter only — never bare-runner python. Fail loud.
set -euo pipefail

V="${1:?python version arg required (3.11/3.12/3.13)}"
VER="${2:?release version required (without leading v)}"
VENV="${SIF_VENV:-/opt/venv-$V}"
PY="$VENV/bin/python"
test -x "$PY" || {
    echo "::error::baked python missing in $VENV — rebuild the SIF: scitex-container apptainer build ci-cpu"
    exit 1
}

shopt -s nullglob
FILES=(dist/*)
[ "${#FILES[@]}" -gt 0 ] || { echo "::error::dist/ empty"; exit 1; }
FAIL=0
for f in "${FILES[@]}"; do
    case "$f" in
        *.whl) META="$("$PY" -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); n=[x for x in z.namelist() if x.endswith(".dist-info/METADATA")][0]; print(z.read(n).decode())' "$f")" ;;
        *.tar.gz) META="$("$PY" -c 'import sys,tarfile; t=tarfile.open(sys.argv[1]); n=[x for x in t.getnames() if x.endswith("PKG-INFO")][0]; print(t.extractfile(n).read().decode())' "$f")" ;;
        *) echo "::error::unexpected artifact $f"; exit 1 ;;
    esac
    NAME="$(printf '%s' "$META" | grep -m1 '^Name: ' | cut -d' ' -f2 | tr -d '\r')"
    MV="$(printf '%s' "$META" | grep -m1 '^Version: ' | cut -d' ' -f2 | tr -d '\r')"
    if [ "$NAME" != "scitex-genai" ] || [ "$MV" != "$VER" ]; then
        echo "::error::$f metadata Name='$NAME' Version='$MV' (want scitex-genai/$VER)"
        FAIL=1
    fi
done
exit "$FAIL"
