#!/usr/bin/env bash
# Qualify downloaded release artifacts BEFORE any OIDC token mint.
# Usage: qualify-dist.sh <version-without-v>
# Asserts every dist/* wheel/sdist carries Name scitex-genai + matching Version
# from its own METADATA/PKG-INFO (not the filename). stdlib only. Fail loud.
set -euo pipefail
VER="${1:?version required}"
FAIL=0
shopt -s nullglob
FILES=(dist/*)
[ "${#FILES[@]}" -gt 0 ] || { echo "::error::dist/ empty"; exit 1; }
for f in "${FILES[@]}"; do
    case "$f" in
        *.whl) META="$(python3 -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); n=[x for x in z.namelist() if x.endswith(".dist-info/METADATA")][0]; print(z.read(n).decode())' "$f")" ;;
        *.tar.gz) META="$(python3 -c 'import sys,tarfile; t=tarfile.open(sys.argv[1]); n=[x for x in t.getnames() if x.endswith("PKG-INFO")][0]; print(t.extractfile(n).read().decode())' "$f")" ;;
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
