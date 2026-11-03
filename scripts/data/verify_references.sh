#!/bin/bash
# Verify reference inputs against the pinned checksums.
#
#   references.sha256  local inputs the analyses read (repo-relative paths)
#   references.md5     GENCODE v48 downloads, from the publisher's MD5SUMS
#
# A file that is absent is reported and skipped (the GENCODE files are only
# present after fetch_references.sh); a file that is present but differs is an
# error.  Exit status is non-zero on any mismatch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_EXTERNAL="${DATA_EXTERNAL:-${ROOT}/data/external}"

hash_of() {  # algo file
    case "$1" in
        sha256) shasum -a 256 "$2" | cut -d' ' -f1 ;;
        md5) if command -v md5sum >/dev/null; then md5sum "$2" | cut -d' ' -f1; else md5 -q "$2"; fi ;;
    esac
}

fail=0
check_list() {  # algo manifest base_dir
    local algo="$1" manifest="$2" base="$3" want path got
    while read -r want path; do
        [[ -z "${want}" || "${want}" == \#* ]] && continue
        if [[ ! -e "${base}/${path}" ]]; then
            echo "[verify] absent, skipped: ${path}"
            continue
        fi
        got="$(hash_of "${algo}" "${base}/${path}")"
        if [[ "${got}" == "${want}" ]]; then
            echo "[verify] ok: ${path}"
        else
            echo "[verify] MISMATCH: ${path} (expected ${want}, got ${got})" >&2
            fail=1
        fi
    done < "${manifest}"
}

check_list sha256 "${HERE}/references.sha256" "${ROOT}"
check_list md5 "${HERE}/references.md5" "${DATA_EXTERNAL}"
exit "${fail}"
