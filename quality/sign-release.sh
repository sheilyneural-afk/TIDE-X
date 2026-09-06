#!/usr/bin/env bash
set -euo pipefail

umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

usage() {
    echo 'usage: TIDEX_RELEASE_GPG_KEY=<40-hex-fingerprint> quality/sign-release.sh <release-directory> [release-archive]' >&2
}

fail() {
    printf 'release signing rejected: %s\n' "$1" >&2
    exit 2
}

[[ $# -eq 1 || $# -eq 2 ]] || { usage; exit 2; }
command -v gpg >/dev/null 2>&1 || fail 'gpg_not_available'
command -v sha256sum >/dev/null 2>&1 || fail 'sha256sum_not_available'
command -v python3 >/dev/null 2>&1 || fail 'python3_not_available'

RELEASE_INPUT=$1
[[ -d "$RELEASE_INPUT" && ! -L "$RELEASE_INPUT" ]] || fail 'release_directory_invalid'
RELEASE_DIR=$(cd -- "$RELEASE_INPUT" && pwd -P)

case "$RELEASE_DIR/" in
    "$QUALITY_ROOT/"*) fail 'release_directory_inside_checkout' ;;
esac

python3 - "$RELEASE_DIR" <<'PY' || exit 2
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
parts = path.parts
current = Path(parts[0])
for part in parts[1:]:
    current /= part
    mode = os.lstat(current).st_mode
    if stat.S_ISLNK(mode):
        print(f"release signing rejected: symlink_path_component:{current}", file=sys.stderr)
        raise SystemExit(2)
if not stat.S_ISDIR(os.lstat(path).st_mode):
    print("release signing rejected: release_directory_not_real_directory", file=sys.stderr)
    raise SystemExit(2)
PY

MANIFEST="$RELEASE_DIR/release-manifest.json"
SUMS="$RELEASE_DIR/SHA256SUMS"
for path in "$MANIFEST" "$SUMS"; do
    [[ -f "$path" && ! -L "$path" ]] || fail "required_regular_file_missing:$(basename "$path")"
done

RELEASE_BASENAME=$(basename "$RELEASE_DIR")
python3 - "$MANIFEST" "$RELEASE_BASENAME" <<'PY' || exit 2
import json
import sys
path, expected = sys.argv[1:]
try:
    manifest = json.load(open(path, encoding='utf-8'))
except Exception as error:
    print(f"release signing rejected: release_manifest_invalid:{error}", file=sys.stderr)
    raise SystemExit(2)
if manifest.get('schema') != 'cerebro.tidex.release_manifest/v1':
    print('release signing rejected: release_manifest_schema_invalid', file=sys.stderr)
    raise SystemExit(2)
if manifest.get('release_id') != expected:
    print('release signing rejected: release_manifest_id_mismatch', file=sys.stderr)
    raise SystemExit(2)
PY

python3 - "$SUMS" "$RELEASE_DIR" <<'PY' || exit 2
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

sums = Path(sys.argv[1])
root = Path(sys.argv[2])
lines = sums.read_text(encoding="utf-8").splitlines()
if not lines:
    print("release signing rejected: checksums_empty", file=sys.stderr)
    raise SystemExit(2)
seen = set()
pattern = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._/-]*)$")
for line in lines:
    match = pattern.fullmatch(line)
    if not match:
        print("release signing rejected: checksums_noncanonical", file=sys.stderr)
        raise SystemExit(2)
    relative = PurePosixPath(match.group(2))
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        print("release signing rejected: checksum_path_escape", file=sys.stderr)
        raise SystemExit(2)
    name = relative.as_posix()
    if name in seen:
        print("release signing rejected: checksum_path_duplicate", file=sys.stderr)
        raise SystemExit(2)
    seen.add(name)
    candidate = root.joinpath(*relative.parts)
    try:
        metadata = os.lstat(candidate)
    except FileNotFoundError:
        print(f"release signing rejected: checksum_subject_missing:{name}", file=sys.stderr)
        raise SystemExit(2)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        print(f"release signing rejected: checksum_subject_not_regular:{name}", file=sys.stderr)
        raise SystemExit(2)
if "release-manifest.json" not in seen:
    print("release signing rejected: release_manifest_not_checksummed", file=sys.stderr)
    raise SystemExit(2)
PY

(
    cd "$RELEASE_DIR"
    sha256sum --strict -c SHA256SUMS >/dev/null
) || fail 'checksum_verification_failed'

ARCHIVE=''
ARCHIVE_SUM=''
if [[ $# -eq 2 ]]; then
    ARCHIVE_INPUT=$2
    [[ -f "$ARCHIVE_INPUT" && ! -L "$ARCHIVE_INPUT" ]] || fail 'release_archive_invalid'
    ARCHIVE=$(cd -- "$(dirname -- "$ARCHIVE_INPUT")" && pwd -P)/$(basename -- "$ARCHIVE_INPUT")
    case "$ARCHIVE" in
        "$QUALITY_ROOT"/*) fail 'release_archive_inside_checkout' ;;
    esac
    [[ "$(dirname -- "$ARCHIVE")" == "$(dirname -- "$RELEASE_DIR")" ]] || fail 'release_archive_not_sibling_of_release_directory'
    [[ "$(basename -- "$ARCHIVE")" == "${RELEASE_BASENAME}.tar.zst" ]] || fail 'release_archive_name_mismatch'
    ARCHIVE_SUM="${ARCHIVE}.sha256"
    [[ -f "$ARCHIVE_SUM" && ! -L "$ARCHIVE_SUM" ]] || fail 'release_archive_checksum_missing'
    (
        cd "$(dirname -- "$ARCHIVE")"
        sha256sum --strict -c "$(basename -- "$ARCHIVE_SUM")" >/dev/null
    ) || fail 'release_archive_checksum_invalid'
fi

KEY=${TIDEX_RELEASE_GPG_KEY:-}
[[ "$KEY" =~ ^[0-9A-Fa-f]{40}$ ]] || fail 'TIDEX_RELEASE_GPG_KEY_must_be_full_40_hex_fingerprint'
KEY=${KEY^^}

SECRET_MATCH=$(gpg --batch --with-colons --fingerprint --list-secret-keys "$KEY" 2>/dev/null \
    | awk -F: '
        $1 == "sec" || $1 == "ssb" { capabilities=$12; next }
        $1 == "fpr" { print toupper($10) " " capabilities }
      ' \
    | awk -v expected="$KEY" '$1 == expected { print; exit }')
[[ -n "$SECRET_MATCH" ]] || fail 'authorized_secret_key_not_found'
KEY_CAPABILITIES=${SECRET_MATCH#* }
[[ "$KEY_CAPABILITIES" == *s* || "$KEY_CAPABILITIES" == *S* ]] || fail 'selected_secret_key_not_signing_capable'

GPG_SECRET_ARGS=(--batch --yes --local-user "${KEY}!")
if [[ -n ${TIDEX_RELEASE_GPG_PASSPHRASE_FILE:-} ]]; then
    PASSPHRASE_FILE=$TIDEX_RELEASE_GPG_PASSPHRASE_FILE
    [[ -f "$PASSPHRASE_FILE" && ! -L "$PASSPHRASE_FILE" ]] || fail 'passphrase_file_invalid'
    PASSPHRASE_FILE=$(cd -- "$(dirname -- "$PASSPHRASE_FILE")" && pwd -P)/$(basename -- "$PASSPHRASE_FILE")
    case "$PASSPHRASE_FILE" in
        "$QUALITY_ROOT"/*) fail 'passphrase_file_inside_checkout' ;;
    esac
    mode=$(stat -c '%a' "$PASSPHRASE_FILE")
    [[ "$mode" == 600 || "$mode" == 400 ]] || fail 'passphrase_file_permissions_must_be_0600_or_0400'
    GPG_SECRET_ARGS+=(--pinentry-mode loopback --passphrase-file "$PASSPHRASE_FILE")
fi

verify_signature() {
    local signature=$1 subject=$2 valid
    valid=$(gpg --batch --status-fd=1 --verify "$signature" "$subject" 2>/dev/null \
        | awk '/^\[GNUPG:\] VALIDSIG / { print toupper($3); exit }')
    [[ "$valid" == "$KEY" ]] || fail "signature_verification_failed:$(basename "$subject")"
}

sign_subject() {
    local subject=$1
    local signature="${subject}.asc"
    local temporary
    if [[ -e "$signature" ]]; then
        [[ -f "$signature" && ! -L "$signature" ]] || fail "existing_signature_not_regular:$(basename "$signature")"
        verify_signature "$signature" "$subject"
        return 0
    fi

    temporary=$(mktemp "$(dirname -- "$subject")/.signature.XXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    gpg "${GPG_SECRET_ARGS[@]}" --armor --detach-sign --output "$temporary" "$subject"
    verify_signature "$temporary" "$subject"
    chmod 0644 "$temporary"
    ln -- "$temporary" "$signature" || fail "signature_publish_collision:$(basename "$signature")"
    rm -f -- "$temporary"
    trap - EXIT
}

sign_subject "$SUMS"
sign_subject "$MANIFEST"
if [[ -n "$ARCHIVE" ]]; then
    sign_subject "$ARCHIVE"
    sign_subject "$ARCHIVE_SUM"
fi

printf 'SIGNED fingerprint=%s checksums=%s manifest=%s' \
    "$KEY" "$(basename "${SUMS}.asc")" "$(basename "${MANIFEST}.asc")"
if [[ -n "$ARCHIVE" ]]; then
    printf ' archive=%s archive_checksum=%s' \
        "$(basename "${ARCHIVE}.asc")" "$(basename "${ARCHIVE_SUM}.asc")"
fi
printf '\n'
