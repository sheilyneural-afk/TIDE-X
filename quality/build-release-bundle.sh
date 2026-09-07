#!/usr/bin/env bash
set -euo pipefail

umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
BINARIES=(
    cerebro-tidex
    acquire-system
    adaptive-learning-cycle
    autonomous-learning-plan
    ledger-diagnose
    pure-linear-runner
    record-representation-evidence
    tidex-finalize
    tidex
)

fail() {
    printf 'release build rejected: %s\n' "$1" >&2
    exit 2
}

[[ $# -eq 1 ]] || fail 'usage: quality/build-release-bundle.sh <output-parent>'
for tool in cargo rustc git python3 sha256sum syft tar zstd flock; do
    command -v "$tool" >/dev/null 2>&1 || fail "required_tool_missing:$tool"
done

OUTPUT_INPUT=$1
[[ -d "$OUTPUT_INPUT" && ! -L "$OUTPUT_INPUT" ]] || fail 'output_parent_invalid'
OUTPUT_PARENT=$(cd -- "$OUTPUT_INPUT" && pwd -P)
case "$OUTPUT_PARENT/" in
    "$QUALITY_ROOT/"*) fail 'output_parent_inside_checkout' ;;
esac

cd "$QUALITY_ROOT"
[[ -z "$(git status --porcelain=v1)" ]] || fail 'working_tree_not_clean'
git diff --check

EXPECTED_STABLE=$(sed -n 's/^channel = "\([^"]*\)"$/\1/p' rust-toolchain.toml)
ACTUAL_STABLE=$(rustc -Vv | sed -n 's/^release: //p')
[[ -n "$EXPECTED_STABLE" && "$ACTUAL_STABLE" == "$EXPECTED_STABLE" ]] || fail 'stable_toolchain_drift'

HEAD_COMMIT=$(git rev-parse HEAD)
HEAD_TREE=$(git rev-parse 'HEAD^{tree}')
SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD)
[[ "$HEAD_COMMIT" =~ ^[0-9a-f]{40}$ && "$HEAD_TREE" =~ ^[0-9a-f]{40}$ ]] || fail 'git_identity_invalid'
[[ "$SOURCE_DATE_EPOCH" =~ ^[0-9]+$ ]] || fail 'source_date_epoch_invalid'

readarray -t PACKAGE_META < <(python3 - <<'PY'
import tomllib
with open('Cargo.toml','rb') as handle:
    package=tomllib.load(handle)['package']
print(package['name'])
print(package['version'])
print('true' if bool(package.get('license') or package.get('license-file')) else 'false')
print(package.get('license',''))
print(package.get('license-file',''))
PY
)
PACKAGE_NAME=${PACKAGE_META[0]}
PACKAGE_VERSION=${PACKAGE_META[1]}
LICENSE_DECLARED=${PACKAGE_META[2]}
PACKAGE_LICENSE=${PACKAGE_META[3]}
PACKAGE_LICENSE_FILE=${PACKAGE_META[4]}
[[ "$PACKAGE_NAME" == cerebro-tidex ]] || fail 'unexpected_package_name'
[[ "$PACKAGE_VERSION" =~ ^[0-9A-Za-z][0-9A-Za-z.+-]*$ ]] || fail 'package_version_not_release_safe'

TARGET=$(rustc -vV | sed -n 's/^host: //p')
[[ "$TARGET" =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'target_not_release_safe'
RELEASE_ID="${PACKAGE_NAME}-${PACKAGE_VERSION}-${TARGET}-${HEAD_COMMIT:0:12}"
FINAL_CONTAINER="$OUTPUT_PARENT/${RELEASE_ID}.release"

WORK=$(mktemp -d /tmp/tidex-release-build.XXXXXX)
cleanup() {
    case "$WORK" in
        /tmp/tidex-release-build.*) rm -rf -- "$WORK" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT

for slot in a b; do
    SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    CARGO_TARGET_DIR="$WORK/target-$slot" \
    CARGO_TERM_COLOR=never \
        cargo build --release --bins --offline --locked
done

for binary in "${BINARIES[@]}"; do
    first="$WORK/target-a/release/$binary"
    second="$WORK/target-b/release/$binary"
    [[ -f "$first" && -x "$first" && -f "$second" && -x "$second" ]] || fail "release_binary_missing:$binary"
    cmp "$first" "$second" || fail "release_binary_not_reproducible:$binary"
done

mapfile -t SOURCE_DIGESTS < <(
    grep -h '^cargo:rustc-env=TIDEX_SOURCE_TREE_DIGEST=' \
        "$WORK"/target-a/release/build/cerebro-tidex-*/output \
        | sed 's/^cargo:rustc-env=TIDEX_SOURCE_TREE_DIGEST=//' \
        | LC_ALL=C sort -u
)
[[ ${#SOURCE_DIGESTS[@]} -eq 1 && "${SOURCE_DIGESTS[0]}" =~ ^[0-9a-f]{64}$ ]] || fail 'compiled_input_digest_ambiguous'
SOURCE_TREE_DIGEST=${SOURCE_DIGESTS[0]}

RELEASE_DIR="$WORK/$RELEASE_ID"
install -d -m 755 "$RELEASE_DIR/bin"
for binary in "${BINARIES[@]}"; do
    install -m 755 "$WORK/target-a/release/$binary" "$RELEASE_DIR/bin/$binary"
done

SYFT_VERSION=$(syft version 2>/dev/null | sed -n 's/^Version:[[:space:]]*//p' | head -1)
[[ -n "$SYFT_VERSION" ]] || fail 'syft_version_unknown'
SBOM_RAW="$WORK/sbom.raw.json"
syft "dir:$QUALITY_ROOT" \
    --source-name "$PACKAGE_NAME" \
    --source-version "$PACKAGE_VERSION" \
    --exclude './.git/**' \
    --exclude './.agent-cache/**' \
    --exclude './.kilo/**' \
    --exclude './.kilocode/**' \
    --exclude './fuzz/**' \
    --exclude './quality/**' \
    --exclude './target/**' \
    -q -o "spdx-json=$SBOM_RAW"

python3 - "$SBOM_RAW" "$RELEASE_DIR/SBOM.spdx.json" "$RELEASE_ID" "$HEAD_COMMIT" "$TARGET" "$SOURCE_DATE_EPOCH" <<'PY'
import datetime as dt
import json
import sys
import uuid

src,dst,name,commit,target,epoch=sys.argv[1:]
doc=json.load(open(src,encoding='utf-8'))
if doc.get('spdxVersion') != 'SPDX-2.3':
    raise SystemExit('release build rejected: sbom_spdx_version_invalid')
if any(pkg.get('name') == 'cerebro-tidex-fuzz' for pkg in doc.get('packages',[])):
    raise SystemExit('release build rejected: sbom_contains_fuzz_package')
doc['name']=name
doc['documentNamespace']='urn:uuid:'+str(uuid.uuid5(uuid.NAMESPACE_URL,f'cerebro-tidex:{commit}:{target}'))
doc.setdefault('creationInfo',{})['created']=dt.datetime.fromtimestamp(int(epoch),dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
for key in ('packages','files','relationships','externalDocumentRefs','snippets','annotations'):
    value=doc.get(key)
    if isinstance(value,list):
        value.sort(key=lambda item: json.dumps(item,sort_keys=True,separators=(',',':')))
with open(dst,'w',encoding='utf-8',newline='\n') as handle:
    json.dump(doc,handle,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    handle.write('\n')
PY
chmod 0644 "$RELEASE_DIR/SBOM.spdx.json"
SBOM_SHA256=$(sha256sum "$RELEASE_DIR/SBOM.spdx.json" | awk '{print $1}')

RUSTC_RELEASE=$(rustc -Vv | sed -n 's/^release: //p')
RUSTC_COMMIT=$(rustc -Vv | sed -n 's/^commit-hash: //p')
RUSTC_COMMIT_DATE=$(rustc -Vv | sed -n 's/^commit-date: //p')
LLVM_VERSION=$(rustc -Vv | sed -n 's/^LLVM version: //p')
CARGO_LOCK_SHA256=$(sha256sum Cargo.lock | awk '{print $1}')
CARGO_TOML_SHA256=$(sha256sum Cargo.toml | awk '{print $1}')
BUILDER_SHA256=$(sha256sum quality/build-release-bundle.sh | awk '{print $1}')
SIGNER_SHA256=$(sha256sum quality/sign-release.sh | awk '{print $1}')
VERIFIER_SHA256=$(sha256sum quality/verify-release.sh | awk '{print $1}')
INSTALL_MANAGER_SHA256=$(sha256sum quality/manage-release-installation.sh | awk '{print $1}')
TAR_VERSION=$(tar --version | head -1)
ZSTD_VERSION=$(zstd --version | head -1)

python3 - \
    "$RELEASE_DIR/release-manifest.json" \
    "$PACKAGE_NAME" "$PACKAGE_VERSION" "$TARGET" \
    "$HEAD_COMMIT" "$HEAD_TREE" "$SOURCE_DATE_EPOCH" "$SOURCE_TREE_DIGEST" \
    "$RUSTC_RELEASE" "$RUSTC_COMMIT" "$RUSTC_COMMIT_DATE" "$LLVM_VERSION" \
    "$CARGO_LOCK_SHA256" "$CARGO_TOML_SHA256" "$SYFT_VERSION" "$SBOM_SHA256" \
    "$BUILDER_SHA256" "$SIGNER_SHA256" "$VERIFIER_SHA256" "$INSTALL_MANAGER_SHA256" \
    "$LICENSE_DECLARED" "$PACKAGE_LICENSE" "$PACKAGE_LICENSE_FILE" "$TAR_VERSION" "$ZSTD_VERSION" \
    "${BINARIES[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    out,package_name,version,target,commit,tree,epoch,source_digest,
    rustc_release,rustc_commit,rustc_date,llvm_version,cargo_lock_sha,cargo_toml_sha,
    syft_version,sbom_sha,builder_sha,signer_sha,verifier_sha,install_manager_sha,
    license_declared,license_expr,license_file,tar_version,zstd_version,*binary_names
)=sys.argv[1:]
root=Path(out).parent
binaries=[]
for name in binary_names:
    path=root/'bin'/name
    data=path.read_bytes()
    binaries.append({
        'name':name,
        'path':f'bin/{name}',
        'sha256':hashlib.sha256(data).hexdigest(),
        'size_bytes':len(data),
    })
manifest={
    'schema':'cerebro.tidex.release_manifest/v1',
    'package':package_name,
    'version':version,
    'release_id':root.name,
    'target':target,
    'git':{'commit':commit,'tree':tree},
    'source_date_epoch':int(epoch),
    'compiled_input_digest':source_digest,
    'build':{
        'profile':'release',
        'offline':True,
        'locked':True,
        'independent_builds':2,
        'binary_match':'bit-for-bit',
    },
    'toolchain':{
        'rustc_release':rustc_release,
        'rustc_commit':rustc_commit,
        'rustc_commit_date':rustc_date,
        'llvm_version':llvm_version,
    },
    'inputs':{
        'Cargo.lock':cargo_lock_sha,
        'Cargo.toml':cargo_toml_sha,
    },
    'sbom':{
        'format':'SPDX-2.3-json',
        'path':'SBOM.spdx.json',
        'sha256':sbom_sha,
        'generator':f'syft-{syft_version}',
    },
    'binaries':binaries,
    'release_tooling':{
        'builder_sha256':builder_sha,
        'signer_sha256':signer_sha,
        'verifier_sha256':verifier_sha,
        'install_manager_sha256':install_manager_sha,
        'tar':tar_version,
        'zstd':zstd_version,
    },
    'signature_policy':{
        'scheme':'OpenPGP detached ASCII-armored',
        'required_for_public_release':True,
        'key_selection':'explicit full fingerprint via TIDEX_RELEASE_GPG_KEY',
    },
    'license':{
        'declared':license_declared == 'true',
        'expression':license_expr or None,
        'file':license_file or None,
    },
}
with open(out,'w',encoding='utf-8',newline='\n') as handle:
    json.dump(manifest,handle,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    handle.write('\n')
PY
chmod 0644 "$RELEASE_DIR/release-manifest.json"

python3 - "$RELEASE_DIR" "${BINARIES[@]}" <<'PY'
import hashlib
import sys
from pathlib import Path
root=Path(sys.argv[1])
names=sys.argv[2:]
subjects=[Path('SBOM.spdx.json'),Path('release-manifest.json')]+[Path('bin')/name for name in names]
subjects=sorted(subjects,key=lambda p:p.as_posix())
with open(root/'SHA256SUMS','w',encoding='utf-8',newline='\n') as handle:
    for relative in subjects:
        data=(root/relative).read_bytes()
        handle.write(f'{hashlib.sha256(data).hexdigest()}  {relative.as_posix()}\n')
PY
chmod 0644 "$RELEASE_DIR/SHA256SUMS"
(
    cd "$RELEASE_DIR"
    sha256sum --strict -c SHA256SUMS >/dev/null
)

TAR_TMP="$WORK/${RELEASE_ID}.tar"
ARCHIVE_TMP="$WORK/${RELEASE_ID}.tar.zst"
tar --sort=name --format=gnu --mtime="@$SOURCE_DATE_EPOCH" \
    --owner=0 --group=0 --numeric-owner \
    -C "$WORK" -cf "$TAR_TMP" "$RELEASE_ID"
zstd -q -19 -T1 -f "$TAR_TMP" -o "$ARCHIVE_TMP"
ARCHIVE_SHA256=$(sha256sum "$ARCHIVE_TMP" | awk '{print $1}')
printf '%s  %s\n' "$ARCHIVE_SHA256" "${RELEASE_ID}.tar.zst" > "$WORK/${RELEASE_ID}.tar.zst.sha256"
chmod 0644 "$WORK/${RELEASE_ID}.tar.zst.sha256"

# Publish the complete release container as one directory rename. The archive is
# the transport form; the unpacked directory is the install/verification form.
exec 9>"$OUTPUT_PARENT/.tidex-release-build.lock"
chmod 0600 "$OUTPUT_PARENT/.tidex-release-build.lock"
flock -x 9
[[ ! -e "$FINAL_CONTAINER" ]] || fail 'release_container_already_exists'
STAGE_CONTAINER=$(mktemp -d "$OUTPUT_PARENT/.${RELEASE_ID}.release.XXXXXX")
cp -a -- "$RELEASE_DIR" "$ARCHIVE_TMP" "$WORK/${RELEASE_ID}.tar.zst.sha256" "$STAGE_CONTAINER/"
(
    cd "$STAGE_CONTAINER/$RELEASE_ID"
    sha256sum --strict -c SHA256SUMS >/dev/null
)
(
    cd "$STAGE_CONTAINER"
    sha256sum --strict -c "${RELEASE_ID}.tar.zst.sha256" >/dev/null
)
mv -T -- "$STAGE_CONTAINER" "$FINAL_CONTAINER"
flock -u 9

printf 'RELEASE_ID=%s\n' "$RELEASE_ID"
printf 'RELEASE_CONTAINER=%s\n' "$FINAL_CONTAINER"
printf 'RELEASE_DIR=%s/%s\n' "$FINAL_CONTAINER" "$RELEASE_ID"
printf 'RELEASE_ARCHIVE=%s/%s.tar.zst\n' "$FINAL_CONTAINER" "$RELEASE_ID"
printf 'RELEASE_ARCHIVE_SHA256=%s\n' "$ARCHIVE_SHA256"
printf 'PUBLIC_LICENSE_DECLARED=%s\n' "$LICENSE_DECLARED"
