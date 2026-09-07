#!/usr/bin/env bash
set -euo pipefail

umask 077
QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
EXPECTED_BINARIES=(
    acquire-system
    adaptive-learning-cycle
    autonomous-learning-plan
    cerebro-tidex
    ledger-diagnose
    pure-linear-runner
    record-representation-evidence
    tidex-finalize
    tidex
)

fail() {
    printf 'release verification rejected: %s\n' "$1" >&2
    exit 2
}

[[ $# -eq 1 || $# -eq 2 ]] || fail 'usage: quality/verify-release.sh <release-directory> [release-archive]'
for tool in python3 sha256sum zstd tar gpg cmp; do
    command -v "$tool" >/dev/null 2>&1 || fail "required_tool_missing:$tool"
done

REQUIRE_SIGNATURE=${TIDEX_RELEASE_REQUIRE_SIGNATURE:-0}
case "$REQUIRE_SIGNATURE" in 0|1) ;; *) fail 'TIDEX_RELEASE_REQUIRE_SIGNATURE_must_be_0_or_1' ;; esac

RELEASE_INPUT=$1
[[ -d "$RELEASE_INPUT" && ! -L "$RELEASE_INPUT" ]] || fail 'release_directory_invalid'
RELEASE_DIR=$(cd -- "$RELEASE_INPUT" && pwd -P)
case "$RELEASE_DIR/" in "$QUALITY_ROOT/"*) fail 'release_directory_inside_checkout' ;; esac

python3 - "$RELEASE_DIR" <<'PY' || exit 2
import os, stat, sys
from pathlib import Path
path=Path(sys.argv[1])
current=Path(path.parts[0])
for part in path.parts[1:]:
    current/=part
    if stat.S_ISLNK(os.lstat(current).st_mode):
        print(f'release verification rejected: symlink_path_component:{current}',file=sys.stderr)
        raise SystemExit(2)
PY

RELEASE_META_RAW=$(python3 - "$RELEASE_DIR" "${EXPECTED_BINARIES[@]}" <<'PY'
import hashlib, json, os, re, stat, sys
from pathlib import Path, PurePosixPath
root=Path(sys.argv[1]); expected_bins=list(sys.argv[2:])
manifest_path=root/'release-manifest.json'; sums_path=root/'SHA256SUMS'; sbom_path=root/'SBOM.spdx.json'
for p in (manifest_path,sums_path,sbom_path):
    try: st=os.lstat(p)
    except FileNotFoundError:
        print(f'release verification rejected: required_file_missing:{p.name}',file=sys.stderr); raise SystemExit(2)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        print(f'release verification rejected: required_file_not_regular:{p.name}',file=sys.stderr); raise SystemExit(2)
try: manifest=json.load(open(manifest_path,encoding='utf-8'))
except Exception as e:
    print(f'release verification rejected: release_manifest_invalid:{e}',file=sys.stderr); raise SystemExit(2)
if manifest.get('schema')!='cerebro.tidex.release_manifest/v1':
    print('release verification rejected: release_manifest_schema_invalid',file=sys.stderr); raise SystemExit(2)
if manifest.get('release_id')!=root.name:
    print('release verification rejected: release_manifest_id_mismatch',file=sys.stderr); raise SystemExit(2)
if manifest.get('package')!='cerebro-tidex':
    print('release verification rejected: release_manifest_package_invalid',file=sys.stderr); raise SystemExit(2)
if not isinstance(manifest.get('target'),str) or not manifest['target']:
    print('release verification rejected: release_manifest_target_invalid',file=sys.stderr); raise SystemExit(2)
entries=manifest.get('binaries')
if not isinstance(entries,list) or len(entries)!=len(expected_bins):
    print('release verification rejected: release_manifest_binary_count_invalid',file=sys.stderr); raise SystemExit(2)
by_name={}
for item in entries:
    if not isinstance(item,dict) or item.get('name') in by_name:
        print('release verification rejected: release_manifest_binary_duplicate',file=sys.stderr); raise SystemExit(2)
    by_name[item.get('name')]=item
if sorted(by_name)!=sorted(expected_bins):
    print('release verification rejected: release_manifest_binary_set_invalid',file=sys.stderr); raise SystemExit(2)

lines=sums_path.read_text(encoding='utf-8').splitlines()
pat=re.compile(r'^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._/-]*)$')
sums={}
for line in lines:
    m=pat.fullmatch(line)
    if not m:
        print('release verification rejected: checksums_noncanonical',file=sys.stderr); raise SystemExit(2)
    rel=PurePosixPath(m.group(2))
    if rel.is_absolute() or '..' in rel.parts or '.' in rel.parts or rel.as_posix() in sums:
        print('release verification rejected: checksum_path_invalid',file=sys.stderr); raise SystemExit(2)
    sums[rel.as_posix()]=m.group(1)
expected_subjects={'SBOM.spdx.json','release-manifest.json'}|{f'bin/{name}' for name in expected_bins}
if set(sums)!=expected_subjects:
    print('release verification rejected: checksum_subject_set_invalid',file=sys.stderr); raise SystemExit(2)
allowed_files=expected_subjects|{'SHA256SUMS','SHA256SUMS.asc','release-manifest.json.asc'}
allowed_dirs={'bin'}
root_mode=stat.S_IMODE(os.lstat(root).st_mode)
if root_mode & 0o022:
    print(f'release verification rejected: release_root_writable_by_others:{root_mode:o}',file=sys.stderr); raise SystemExit(2)
for candidate in root.rglob('*'):
    rel=candidate.relative_to(root).as_posix()
    st=os.lstat(candidate)
    if stat.S_ISLNK(st.st_mode):
        print(f'release verification rejected: release_tree_symlink:{rel}',file=sys.stderr); raise SystemExit(2)
    if stat.S_ISDIR(st.st_mode):
        if rel not in allowed_dirs:
            print(f'release verification rejected: release_tree_extra_directory:{rel}',file=sys.stderr); raise SystemExit(2)
        if stat.S_IMODE(st.st_mode) & 0o022:
            print(f'release verification rejected: release_directory_writable_by_others:{rel}',file=sys.stderr); raise SystemExit(2)
    elif stat.S_ISREG(st.st_mode):
        if rel not in allowed_files:
            print(f'release verification rejected: release_tree_extra_file:{rel}',file=sys.stderr); raise SystemExit(2)
        if stat.S_IMODE(st.st_mode) & 0o022:
            print(f'release verification rejected: release_file_writable_by_others:{rel}',file=sys.stderr); raise SystemExit(2)
        if rel.startswith('bin/') and not (stat.S_IMODE(st.st_mode) & 0o111):
            print(f'release verification rejected: release_binary_not_executable:{rel}',file=sys.stderr); raise SystemExit(2)
    else:
        print(f'release verification rejected: release_tree_special_file:{rel}',file=sys.stderr); raise SystemExit(2)
for rel,digest in sums.items():
    p=root.joinpath(*PurePosixPath(rel).parts)
    try: st=os.lstat(p)
    except FileNotFoundError:
        print(f'release verification rejected: checksum_subject_missing:{rel}',file=sys.stderr); raise SystemExit(2)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        print(f'release verification rejected: checksum_subject_not_regular:{rel}',file=sys.stderr); raise SystemExit(2)
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual!=digest:
        print(f'release verification rejected: checksum_mismatch:{rel}',file=sys.stderr); raise SystemExit(2)
for name,item in by_name.items():
    rel=f'bin/{name}'; p=root/rel
    if item.get('path')!=rel or item.get('sha256')!=sums[rel] or item.get('size_bytes')!=p.stat().st_size:
        print(f'release verification rejected: release_manifest_binary_binding_invalid:{name}',file=sys.stderr); raise SystemExit(2)
try: sbom=json.load(open(sbom_path,encoding='utf-8'))
except Exception as e:
    print(f'release verification rejected: sbom_invalid:{e}',file=sys.stderr); raise SystemExit(2)
if sbom.get('spdxVersion')!='SPDX-2.3' or sbom.get('name')!=root.name:
    print('release verification rejected: sbom_identity_invalid',file=sys.stderr); raise SystemExit(2)
if manifest.get('sbom',{}).get('sha256')!=sums['SBOM.spdx.json']:
    print('release verification rejected: sbom_manifest_binding_invalid',file=sys.stderr); raise SystemExit(2)
print(root.name)
print(manifest['target'])
print('true' if manifest.get('license',{}).get('declared') is True else 'false')
for rel in sorted(expected_subjects): print(rel)
PY
) || exit 2
mapfile -t RELEASE_META <<< "$RELEASE_META_RAW"
[[ ${#RELEASE_META[@]} -ge 13 ]] || fail 'release_metadata_incomplete'
RELEASE_ID=${RELEASE_META[0]}
RELEASE_TARGET=${RELEASE_META[1]}
LICENSE_DECLARED=${RELEASE_META[2]}
CORE_SUBJECTS=("${RELEASE_META[@]:3}")

ARCHIVE=''
ARCHIVE_SUM=''
if [[ $# -eq 2 ]]; then
    ARCHIVE_INPUT=$2
    [[ -f "$ARCHIVE_INPUT" && ! -L "$ARCHIVE_INPUT" ]] || fail 'release_archive_invalid'
    ARCHIVE=$(cd -- "$(dirname -- "$ARCHIVE_INPUT")" && pwd -P)/$(basename -- "$ARCHIVE_INPUT")
    case "$ARCHIVE" in "$QUALITY_ROOT"/*) fail 'release_archive_inside_checkout' ;; esac
    [[ "$(dirname -- "$ARCHIVE")" == "$(dirname -- "$RELEASE_DIR")" ]] || fail 'release_archive_not_sibling_of_release_directory'
    [[ "$(basename -- "$ARCHIVE")" == "${RELEASE_ID}.tar.zst" ]] || fail 'release_archive_name_mismatch'
    ARCHIVE_SUM="${ARCHIVE}.sha256"
    [[ -f "$ARCHIVE_SUM" && ! -L "$ARCHIVE_SUM" ]] || fail 'release_archive_checksum_missing'
    expected_line="$(sha256sum "$ARCHIVE" | awk '{print $1}')  $(basename -- "$ARCHIVE")"
    [[ "$(cat "$ARCHIVE_SUM")" == "$expected_line" ]] || fail 'release_archive_checksum_invalid'

    LISTING=$(mktemp /tmp/tidex-release-listing.XXXXXX)
    EXTRACT=$(mktemp -d /tmp/tidex-release-extract.XXXXXX)
    cleanup_archive_tmp(){ rm -f -- "$LISTING"; rm -rf -- "$EXTRACT"; }
    trap cleanup_archive_tmp EXIT
    zstd -q -d -c "$ARCHIVE" | tar -tf - > "$LISTING"
    ARCHIVE_SUBJECTS=(SHA256SUMS "${CORE_SUBJECTS[@]}")
    python3 - "$LISTING" "$RELEASE_ID" "${ARCHIVE_SUBJECTS[@]}" <<'PY' || exit 2
import sys
from pathlib import PurePosixPath
listing,release,*subjects=sys.argv[1:]
actual=[line.strip() for line in open(listing,encoding='utf-8') if line.strip()]
expected=[release+'/',release+'/bin/']+[release+'/'+s for s in subjects]
if set(actual)!=set(expected) or len(actual)!=len(expected):
    print('release verification rejected: archive_member_set_invalid',file=sys.stderr); raise SystemExit(2)
for name in actual:
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        print('release verification rejected: archive_path_escape',file=sys.stderr); raise SystemExit(2)
PY
    zstd -q -d -c "$ARCHIVE" | tar -xf - -C "$EXTRACT" --no-same-owner
    for rel in "${ARCHIVE_SUBJECTS[@]}"; do
        cmp "$RELEASE_DIR/$rel" "$EXTRACT/$RELEASE_ID/$rel" || fail "archive_content_mismatch:$rel"
    done
    cleanup_archive_tmp
    trap - EXIT
fi

KEY=${TIDEX_RELEASE_GPG_KEY:-}
if [[ "$REQUIRE_SIGNATURE" -eq 1 && -z "$KEY" ]]; then
    fail 'signature_required_but_TIDEX_RELEASE_GPG_KEY_missing'
fi
SIGNATURE_STATUS=not-requested
if [[ -n "$KEY" ]]; then
    [[ "$KEY" =~ ^[0-9A-Fa-f]{40}$ ]] || fail 'TIDEX_RELEASE_GPG_KEY_must_be_full_40_hex_fingerprint'
    KEY=${KEY^^}
    gpg --batch --list-keys "$KEY" >/dev/null 2>&1 || fail 'trusted_public_key_not_found'
    verify_sig(){
        local subject=$1
        local signature="${subject}.asc"
        local valid
        [[ -f "$signature" && ! -L "$signature" ]] || fail "signature_missing:$(basename "$subject")"
        valid=$(gpg --batch --status-fd=1 --verify "$signature" "$subject" 2>/dev/null | awk '/^\[GNUPG:\] VALIDSIG /{print toupper($3); exit}')
        [[ "$valid" == "$KEY" ]] || fail "signature_invalid:$(basename "$subject")"
    }
    verify_sig "$RELEASE_DIR/SHA256SUMS"
    verify_sig "$RELEASE_DIR/release-manifest.json"
    if [[ -n "$ARCHIVE" ]]; then
        verify_sig "$ARCHIVE"
        verify_sig "$ARCHIVE_SUM"
    fi
    SIGNATURE_STATUS=verified
fi

printf 'VERIFIED release_id=%s target=%s archive=%s signatures=%s license_declared=%s\n' \
    "$RELEASE_ID" "$RELEASE_TARGET" "$([[ -n "$ARCHIVE" ]] && echo verified || echo not-requested)" \
    "$SIGNATURE_STATUS" "$LICENSE_DECLARED"
