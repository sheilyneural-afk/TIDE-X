#!/usr/bin/env bash
set -euo pipefail

umask 077
QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
VERIFIER="$QUALITY_ROOT/quality/verify-release.sh"

usage() {
    echo 'usage: TIDEX_RELEASE_GPG_KEY=<40-hex-fingerprint> quality/sign-release.sh <release-directory> [release-archive]' >&2
}
fail() {
    printf 'release signing rejected: %s\n' "$1" >&2
    exit 2
}

[[ $# -eq 1 || $# -eq 2 ]] || { usage; exit 2; }
[[ -f "$VERIFIER" && ! -L "$VERIFIER" && -x "$VERIFIER" ]] || fail 'release_verifier_unavailable'
command -v gpg >/dev/null 2>&1 || fail 'gpg_not_available'

# Integrity/structure/archive authority is centralized in verify-release.sh.
env -u TIDEX_RELEASE_GPG_KEY TIDEX_RELEASE_REQUIRE_SIGNATURE=0 \
    "$VERIFIER" "$@" >/dev/null

RELEASE_DIR=$(cd -- "$1" && pwd -P)
SUMS="$RELEASE_DIR/SHA256SUMS"
MANIFEST="$RELEASE_DIR/release-manifest.json"
ARCHIVE=''
ARCHIVE_SUM=''
if [[ $# -eq 2 ]]; then
    ARCHIVE=$(cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
    ARCHIVE_SUM="${ARCHIVE}.sha256"
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
    case "$PASSPHRASE_FILE" in "$QUALITY_ROOT"/*) fail 'passphrase_file_inside_checkout' ;; esac
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

# A signing operation is successful only if the shared verifier accepts every
# required signature with the exact selected fingerprint.
TIDEX_RELEASE_GPG_KEY="$KEY" TIDEX_RELEASE_REQUIRE_SIGNATURE=1 \
    "$VERIFIER" "$@" >/dev/null

printf 'SIGNED fingerprint=%s checksums=%s manifest=%s' \
    "$KEY" "$(basename "${SUMS}.asc")" "$(basename "${MANIFEST}.asc")"
if [[ -n "$ARCHIVE" ]]; then
    printf ' archive=%s archive_checksum=%s' \
        "$(basename "${ARCHIVE}.asc")" "$(basename "${ARCHIVE_SUM}.asc")"
fi
printf '\n'
