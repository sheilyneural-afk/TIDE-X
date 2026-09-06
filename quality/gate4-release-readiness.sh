#!/usr/bin/env bash
set -euo pipefail
umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
QUALITY_TMP=$(mktemp -d /tmp/tidex-gate4-release.XXXXXX)
QUALITY_START_EPOCH=$(date +%s)

cleanup() {
    case "$QUALITY_TMP" in
        /tmp/tidex-gate4-release.*) find "$QUALITY_TMP" -depth -delete ;;
        *) echo "P4 cleanup refused unexpected path: $QUALITY_TMP" >&2; return 1 ;;
    esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

fail() { printf 'P4 RECHAZADA: %s\n' "$1" >&2; exit 2; }
sha_file() { sha256sum "$1" | awk '{print $1}'; }

snapshot_checkout() {
    local prefix=$1
    find . -path ./.git -prune -o -type f -printf '%m\t%s\t%p\0' \
        | LC_ALL=C sort -z > "${prefix}.metadata"
    find . -path ./.git -prune -o -type f -print0 \
        | LC_ALL=C sort -z | xargs -0 -r sha256sum --zero > "${prefix}.sha256"
}

resolve_release_container() {
    local parent=$1
    local -a matches=()
    mapfile -t matches < <(find "$parent" -mindepth 1 -maxdepth 1 -type d -name '*.release' -print | LC_ALL=C sort)
    [[ ${#matches[@]} -eq 1 ]] || fail "release_container_count:${#matches[@]}:$parent"
    printf '%s\n' "${matches[0]}"
}

resolve_release_dir() {
    local container=$1
    local -a matches=()
    mapfile -t matches < <(find "$container" -mindepth 1 -maxdepth 1 -type d -name 'cerebro-tidex-*' -print | LC_ALL=C sort)
    [[ ${#matches[@]} -eq 1 ]] || fail "release_directory_count:${#matches[@]}:$container"
    printf '%s\n' "${matches[0]}"
}

resolve_release_archive() {
    local container=$1
    local -a matches=()
    mapfile -t matches < <(find "$container" -mindepth 1 -maxdepth 1 -type f -name '*.tar.zst' -print | LC_ALL=C sort)
    [[ ${#matches[@]} -eq 1 ]] || fail "release_archive_count:${#matches[@]}:$container"
    printf '%s\n' "${matches[0]}"
}

snapshot_release_tree() {
    local root=$1 output=$2
    python3 - "$root" "$output" <<'PY'
import hashlib, os, stat, sys
from pathlib import Path
root=Path(sys.argv[1]); output=Path(sys.argv[2])
rows=[]
for path in sorted(root.rglob('*'), key=lambda p:p.relative_to(root).as_posix()):
    rel=path.relative_to(root).as_posix()
    st=os.lstat(path); mode=stat.S_IMODE(st.st_mode)
    if stat.S_ISDIR(st.st_mode): rows.append((rel,'dir',mode,0,None))
    elif stat.S_ISREG(st.st_mode):
        data=path.read_bytes(); rows.append((rel,'file',mode,len(data),hashlib.sha256(data).hexdigest()))
    elif stat.S_ISLNK(st.st_mode): rows.append((rel,'symlink',mode,0,os.readlink(path)))
    else: rows.append((rel,'special',mode,st.st_size,None))
with open(output,'w',encoding='utf-8',newline='\n') as handle:
    for row in rows:
        handle.write('\t'.join('' if value is None else str(value) for value in row)+'\n')
PY
}

cd "$QUALITY_ROOT"
for tool in git bash python3 cargo rustc sha256sum cmp diff gpg syft tar zstd find flock shellcheck; do
    command -v "$tool" >/dev/null 2>&1 || fail "required_tool_missing:$tool"
done
[[ -z "$(git status --porcelain=v1)" ]] || fail 'working_tree_not_clean'
git diff --check
shellcheck -x \
    quality/build-release-bundle.sh \
    quality/verify-release.sh \
    quality/sign-release.sh \
    quality/manage-release-installation.sh \
    quality/gate4-release-readiness.sh

QUALITY_HEAD=$(git rev-parse HEAD)
QUALITY_PARENT=$(git rev-parse HEAD^)
[[ "$QUALITY_HEAD" =~ ^[0-9a-f]{40}$ && "$QUALITY_PARENT" =~ ^[0-9a-f]{40}$ ]] || fail 'git_identity_invalid'

if [[ -z ${QUALITY_P4_RECEIPT_PATH:-} ]]; then
    QUALITY_P4_RECEIPT_PATH="/tmp/tidex-gate4-release-${QUALITY_HEAD:0:12}.json"
elif [[ "$QUALITY_P4_RECEIPT_PATH" != /* ]]; then
    QUALITY_P4_RECEIPT_PATH="$QUALITY_ROOT/$QUALITY_P4_RECEIPT_PATH"
fi
case "$QUALITY_P4_RECEIPT_PATH" in
    "$QUALITY_ROOT"|"$QUALITY_ROOT"/*) fail 'receipt_must_be_outside_checkout' ;;
esac
mkdir -p "$(dirname "$QUALITY_P4_RECEIPT_PATH")"

snapshot_checkout "$QUALITY_TMP/checkout-before"
QUALITY_CHECKOUT_SHA256=$(sha_file "$QUALITY_TMP/checkout-before.sha256")
QUALITY_METADATA_SHA256=$(sha_file "$QUALITY_TMP/checkout-before.metadata")

# P4 is cumulative. No release-readiness result is accepted without the full P3 chain.
P3_RECEIPT="$QUALITY_TMP/p3-receipt.json"
QUALITY_RECEIPT_PATH="$P3_RECEIPT" bash quality/gate3-assurance.sh 2>&1 | tee "$QUALITY_TMP/p3.log"
sha256sum -c "${P3_RECEIPT}.sha256" >/dev/null
P3_RECEIPT_SHA256=$(sha_file "$P3_RECEIPT")

# Build the current release twice. Each builder run itself also performs two
# independent release builds, so this tests both binary and complete-bundle reproducibility.
install -d -m 700 "$QUALITY_TMP/current-a" "$QUALITY_TMP/current-b" "$QUALITY_TMP/previous-out"
quality/build-release-bundle.sh "$QUALITY_TMP/current-a" | tee "$QUALITY_TMP/current-a.log"
quality/build-release-bundle.sh "$QUALITY_TMP/current-b" | tee "$QUALITY_TMP/current-b.log"
CURRENT_CONTAINER_A=$(resolve_release_container "$QUALITY_TMP/current-a")
CURRENT_CONTAINER_B=$(resolve_release_container "$QUALITY_TMP/current-b")
CURRENT_RELEASE_A=$(resolve_release_dir "$CURRENT_CONTAINER_A")
CURRENT_RELEASE_B=$(resolve_release_dir "$CURRENT_CONTAINER_B")
CURRENT_ARCHIVE_A=$(resolve_release_archive "$CURRENT_CONTAINER_A")
CURRENT_ARCHIVE_B=$(resolve_release_archive "$CURRENT_CONTAINER_B")

snapshot_release_tree "$CURRENT_CONTAINER_A" "$QUALITY_TMP/current-a.inventory"
snapshot_release_tree "$CURRENT_CONTAINER_B" "$QUALITY_TMP/current-b.inventory"
cmp "$QUALITY_TMP/current-a.inventory" "$QUALITY_TMP/current-b.inventory"
diff -qr --no-dereference "$CURRENT_CONTAINER_A" "$CURRENT_CONTAINER_B" >/dev/null
CURRENT_BUNDLE_INVENTORY_SHA256=$(sha_file "$QUALITY_TMP/current-a.inventory")
CURRENT_ARCHIVE_SHA256=$(sha_file "$CURRENT_ARCHIVE_A")
[[ "$CURRENT_ARCHIVE_SHA256" == "$(sha_file "$CURRENT_ARCHIVE_B")" ]] || fail 'current_archive_not_reproducible'
quality/verify-release.sh "$CURRENT_RELEASE_A" "$CURRENT_ARCHIVE_A" > "$QUALITY_TMP/current-a.verify"
quality/verify-release.sh "$CURRENT_RELEASE_B" "$CURRENT_ARCHIVE_B" > "$QUALITY_TMP/current-b.verify"

CURRENT_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_id"])' "$CURRENT_RELEASE_A/release-manifest.json")
[[ "$CURRENT_ID" == "$(basename "$CURRENT_RELEASE_A")" && "$CURRENT_ID" == "$(basename "$CURRENT_RELEASE_B")" ]] || fail 'current_release_id_mismatch'

# Build a second, genuine release identity from the immediate parent in a fully
# separate local clone. No fixtures or relabelled manifests are used for rollback evidence.
PREVIOUS_REPO="$QUALITY_TMP/previous-repo"
git clone --quiet --no-hardlinks --no-checkout "$QUALITY_ROOT" "$PREVIOUS_REPO"
git -C "$PREVIOUS_REPO" checkout --quiet --detach "$QUALITY_PARENT"
[[ -x "$PREVIOUS_REPO/quality/build-release-bundle.sh" ]] || fail 'parent_release_builder_missing'
(
    cd "$PREVIOUS_REPO"
    quality/build-release-bundle.sh "$QUALITY_TMP/previous-out" | tee "$QUALITY_TMP/previous.log"
)
PREVIOUS_CONTAINER=$(resolve_release_container "$QUALITY_TMP/previous-out")
PREVIOUS_RELEASE=$(resolve_release_dir "$PREVIOUS_CONTAINER")
PREVIOUS_ARCHIVE=$(resolve_release_archive "$PREVIOUS_CONTAINER")
quality/verify-release.sh "$PREVIOUS_RELEASE" "$PREVIOUS_ARCHIVE" > "$QUALITY_TMP/previous.verify"
PREVIOUS_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_id"])' "$PREVIOUS_RELEASE/release-manifest.json")
[[ "$PREVIOUS_ID" != "$CURRENT_ID" ]] || fail 'previous_and_current_release_identity_equal'
PREVIOUS_ARCHIVE_SHA256=$(sha_file "$PREVIOUS_ARCHIVE")

# Exercise real OpenPGP signing without inventing a persistent distribution identity.
# This key is test-only, confined to QUALITY_TMP and destroyed by cleanup.
GNUPGHOME="$QUALITY_TMP/gnupg"
export GNUPGHOME
install -d -m 700 "$GNUPGHOME"
gpg --batch --quiet --passphrase '' --quick-generate-key \
    'TIDE-X P4 Ephemeral Gate <tidex-p4-gate@invalid>' rsa2048 sign 1d
P4_TEST_FINGERPRINT=$(gpg --batch --with-colons --fingerprint --list-secret-keys \
    | awk -F: '$1=="fpr" { print toupper($10); exit }')
[[ "$P4_TEST_FINGERPRINT" =~ ^[0-9A-F]{40}$ ]] || fail 'ephemeral_signing_fingerprint_invalid'
TIDEX_RELEASE_GPG_KEY="$P4_TEST_FINGERPRINT" \
    quality/sign-release.sh "$CURRENT_RELEASE_A" "$CURRENT_ARCHIVE_A" > "$QUALITY_TMP/current.sign"
TIDEX_RELEASE_GPG_KEY="$P4_TEST_FINGERPRINT" \
    quality/sign-release.sh "$PREVIOUS_RELEASE" "$PREVIOUS_ARCHIVE" > "$QUALITY_TMP/previous.sign"
TIDEX_RELEASE_GPG_KEY="$P4_TEST_FINGERPRINT" TIDEX_RELEASE_REQUIRE_SIGNATURE=1 \
    quality/verify-release.sh "$CURRENT_RELEASE_A" "$CURRENT_ARCHIVE_A" > "$QUALITY_TMP/current.signed.verify"
TIDEX_RELEASE_GPG_KEY="$P4_TEST_FINGERPRINT" TIDEX_RELEASE_REQUIRE_SIGNATURE=1 \
    quality/verify-release.sh "$PREVIOUS_RELEASE" "$PREVIOUS_ARCHIVE" > "$QUALITY_TMP/previous.signed.verify"

# Tampering with a structurally valid transport archive must fail specifically on its checksum.
TAMPER_CONTAINER="$QUALITY_TMP/tampered-container"
cp -a "$CURRENT_CONTAINER_A" "$TAMPER_CONTAINER"
TAMPER_RELEASE="$TAMPER_CONTAINER/$(basename "$CURRENT_RELEASE_A")"
TAMPER_ARCHIVE="$TAMPER_CONTAINER/$(basename "$CURRENT_ARCHIVE_A")"
printf x >> "$TAMPER_ARCHIVE"
set +e
TIDEX_RELEASE_GPG_KEY="$P4_TEST_FINGERPRINT" TIDEX_RELEASE_REQUIRE_SIGNATURE=1 \
    quality/verify-release.sh "$TAMPER_RELEASE" "$TAMPER_ARCHIVE" \
    > "$QUALITY_TMP/tampered.out" 2> "$QUALITY_TMP/tampered.err"
TAMPER_CODE=$?
set -e
[[ "$TAMPER_CODE" -eq 2 ]] || fail 'tampered_archive_not_rejected'
grep -Fq 'release_archive_checksum_invalid' "$QUALITY_TMP/tampered.err" || fail 'tampered_archive_wrong_rejection_reason'

# Install the previous signed release, upgrade to current, recover a deleted
# derived pointer, rollback, roll forward, uninstall only the inactive release,
# and prove private state survives every transition.
INSTALL_ROOT="$QUALITY_TMP/install-root"
STATE_ROOT="$QUALITY_TMP/private-state"
P4_ENV=(env GNUPGHOME="$GNUPGHOME" TIDEX_RELEASE_GPG_KEY="$P4_TEST_FINGERPRINT" TIDEX_RELEASE_REQUIRE_SIGNATURE=1)
"${P4_ENV[@]}" quality/manage-release-installation.sh install \
    "$PREVIOUS_RELEASE" "$PREVIOUS_ARCHIVE" "$INSTALL_ROOT" "$STATE_ROOT" | tee "$QUALITY_TMP/install-previous.log"
"${P4_ENV[@]}" quality/manage-release-installation.sh activate "$INSTALL_ROOT" "$PREVIOUS_ID" | tee "$QUALITY_TMP/activate-previous.log"
printf 'p4-private-state-must-survive\n' > "$STATE_ROOT/p4-sentinel"
chmod 0600 "$STATE_ROOT/p4-sentinel"
STATE_SENTINEL_SHA256=$(sha_file "$STATE_ROOT/p4-sentinel")

"${P4_ENV[@]}" quality/manage-release-installation.sh install \
    "$CURRENT_RELEASE_A" "$CURRENT_ARCHIVE_A" "$INSTALL_ROOT" "$STATE_ROOT" | tee "$QUALITY_TMP/install-current.log"
"${P4_ENV[@]}" quality/manage-release-installation.sh activate "$INSTALL_ROOT" "$CURRENT_ID" | tee "$QUALITY_TMP/activate-current.log"
"${P4_ENV[@]}" quality/manage-release-installation.sh verify "$INSTALL_ROOT" > "$QUALITY_TMP/verify-upgraded.log"

python3 - "$INSTALL_ROOT/activation.json" "$CURRENT_ID" "$PREVIOUS_ID" <<'PY'
import json,sys
path,active,previous=sys.argv[1:]
value=json.load(open(path,encoding='utf-8'))
expected={'schema':'cerebro.tidex.activation/v1','active':active,'previous':previous,'revision':2}
if value != expected: raise SystemExit(f'P4 activation after upgrade mismatch: {value!r}')
PY
[[ "$(readlink "$INSTALL_ROOT/current")" == "releases/$CURRENT_ID" ]] || fail 'current_pointer_not_current_after_upgrade'
[[ "$(sha_file "$STATE_ROOT/p4-sentinel")" == "$STATE_SENTINEL_SHA256" ]] || fail 'state_changed_during_upgrade'

# Verify does not silently repair a missing derived pointer; activate does, without authority advance.
rm "$INSTALL_ROOT/current"
set +e
"${P4_ENV[@]}" quality/manage-release-installation.sh verify "$INSTALL_ROOT" \
    > "$QUALITY_TMP/missing-current.out" 2> "$QUALITY_TMP/missing-current.err"
MISSING_CURRENT_CODE=$?
set -e
[[ "$MISSING_CURRENT_CODE" -eq 2 ]] || fail 'missing_current_pointer_not_detected'
REVISION_BEFORE_RECOVERY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$INSTALL_ROOT/activation.json")
"${P4_ENV[@]}" quality/manage-release-installation.sh activate "$INSTALL_ROOT" "$CURRENT_ID" > "$QUALITY_TMP/recover-current.log"
REVISION_AFTER_RECOVERY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$INSTALL_ROOT/activation.json")
[[ "$REVISION_BEFORE_RECOVERY" == "$REVISION_AFTER_RECOVERY" ]] || fail 'derived_pointer_recovery_advanced_authority'
[[ "$(readlink "$INSTALL_ROOT/current")" == "releases/$CURRENT_ID" ]] || fail 'derived_pointer_recovery_failed'

# The installed current binary is real and preserves the fail-closed empty-engine contract.
set +e
TIDEX_PRIVATE_ROOT="$STATE_ROOT" "$INSTALL_ROOT/current/bin/cerebro-tidex" status \
    > "$QUALITY_TMP/runtime.out" 2> "$QUALITY_TMP/runtime.err"
RUNTIME_CODE=$?
set -e
[[ "$RUNTIME_CODE" -eq 2 && ! -s "$QUALITY_TMP/runtime.out" ]] || fail 'installed_runtime_empty_state_exit_contract'
grep -Fxq 'integrity:active_skill_bank_missing' "$QUALITY_TMP/runtime.err" || fail 'installed_runtime_empty_state_diagnostic'

"${P4_ENV[@]}" quality/manage-release-installation.sh rollback "$INSTALL_ROOT" | tee "$QUALITY_TMP/rollback.log"
"${P4_ENV[@]}" quality/manage-release-installation.sh verify "$INSTALL_ROOT" > "$QUALITY_TMP/verify-rollback.log"
[[ "$(readlink "$INSTALL_ROOT/current")" == "releases/$PREVIOUS_ID" ]] || fail 'rollback_pointer_mismatch'
[[ "$(sha_file "$STATE_ROOT/p4-sentinel")" == "$STATE_SENTINEL_SHA256" ]] || fail 'state_changed_during_rollback'

"${P4_ENV[@]}" quality/manage-release-installation.sh activate "$INSTALL_ROOT" "$CURRENT_ID" > "$QUALITY_TMP/roll-forward.log"
"${P4_ENV[@]}" quality/manage-release-installation.sh uninstall "$INSTALL_ROOT" "$PREVIOUS_ID" > "$QUALITY_TMP/uninstall-previous.log"
"${P4_ENV[@]}" quality/manage-release-installation.sh verify "$INSTALL_ROOT" > "$QUALITY_TMP/verify-final.log"
[[ ! -e "$INSTALL_ROOT/releases/$PREVIOUS_ID" && -d "$INSTALL_ROOT/releases/$CURRENT_ID" ]] || fail 'inactive_uninstall_layout_invalid'
[[ "$(sha_file "$STATE_ROOT/p4-sentinel")" == "$STATE_SENTINEL_SHA256" ]] || fail 'state_changed_during_uninstall'

set +e
"${P4_ENV[@]}" quality/manage-release-installation.sh rollback "$INSTALL_ROOT" \
    > "$QUALITY_TMP/rollback-missing.out" 2> "$QUALITY_TMP/rollback-missing.err"
ROLLBACK_MISSING_CODE=$?
"${P4_ENV[@]}" quality/manage-release-installation.sh uninstall "$INSTALL_ROOT" "$CURRENT_ID" \
    > "$QUALITY_TMP/uninstall-active.out" 2> "$QUALITY_TMP/uninstall-active.err"
UNINSTALL_ACTIVE_CODE=$?
set -e
[[ "$ROLLBACK_MISSING_CODE" -eq 2 ]] || fail 'rollback_without_previous_not_rejected'
[[ "$UNINSTALL_ACTIVE_CODE" -eq 2 ]] || fail 'active_uninstall_not_rejected'
grep -Fq 'rollback_target_missing' "$QUALITY_TMP/rollback-missing.err" || fail 'rollback_missing_diagnostic_invalid'
grep -Fq 'cannot_uninstall_active_release' "$QUALITY_TMP/uninstall-active.err" || fail 'active_uninstall_diagnostic_invalid'

FINAL_ACTIVATION="$QUALITY_TMP/final-activation.json"
cp "$INSTALL_ROOT/activation.json" "$FINAL_ACTIVATION"
python3 - "$FINAL_ACTIVATION" "$CURRENT_ID" <<'PY'
import json,sys
path,active=sys.argv[1:]
value=json.load(open(path,encoding='utf-8'))
expected={'schema':'cerebro.tidex.activation/v1','active':active,'previous':None,'revision':5}
if value != expected: raise SystemExit(f'P4 final activation mismatch: {value!r}')
PY

snapshot_checkout "$QUALITY_TMP/checkout-after"
cmp "$QUALITY_TMP/checkout-before.metadata" "$QUALITY_TMP/checkout-after.metadata"
cmp "$QUALITY_TMP/checkout-before.sha256" "$QUALITY_TMP/checkout-after.sha256"
git diff --check
[[ -z "$(git status --porcelain=v1)" ]] || fail 'working_tree_changed_during_gate'

QUALITY_END_EPOCH=$(date +%s)
QUALITY_GATE4_SHA256=$(sha_file quality/gate4-release-readiness.sh)
QUALITY_BUILDER_SHA256=$(sha_file quality/build-release-bundle.sh)
QUALITY_VERIFIER_SHA256=$(sha_file quality/verify-release.sh)
QUALITY_SIGNER_SHA256=$(sha_file quality/sign-release.sh)
QUALITY_INSTALLER_SHA256=$(sha_file quality/manage-release-installation.sh)
CURRENT_MANIFEST_SHA256=$(sha_file "$CURRENT_RELEASE_A/release-manifest.json")
CURRENT_SBOM_SHA256=$(sha_file "$CURRENT_RELEASE_A/SBOM.spdx.json")
PREVIOUS_MANIFEST_SHA256=$(sha_file "$PREVIOUS_RELEASE/release-manifest.json")
PREVIOUS_SBOM_SHA256=$(sha_file "$PREVIOUS_RELEASE/SBOM.spdx.json")
LICENSE_DECLARED=$(python3 -c 'import json,sys; print("true" if json.load(open(sys.argv[1]))["license"]["declared"] else "false")' "$CURRENT_RELEASE_A/release-manifest.json")

PUBLIC_BLOCKERS_FILE="$QUALITY_TMP/public-blockers.txt"
: > "$PUBLIC_BLOCKERS_FILE"
[[ "$LICENSE_DECLARED" == true ]] || echo product_license_undeclared >> "$PUBLIC_BLOCKERS_FILE"
echo production_distribution_signature_not_performed >> "$PUBLIC_BLOCKERS_FILE"

python3 - \
    "$QUALITY_P4_RECEIPT_PATH" "$P3_RECEIPT" "$FINAL_ACTIVATION" "$PUBLIC_BLOCKERS_FILE" \
    "$QUALITY_HEAD" "$QUALITY_PARENT" "$QUALITY_CHECKOUT_SHA256" "$QUALITY_METADATA_SHA256" \
    "$QUALITY_START_EPOCH" "$QUALITY_END_EPOCH" "$P3_RECEIPT_SHA256" \
    "$QUALITY_GATE4_SHA256" "$QUALITY_BUILDER_SHA256" "$QUALITY_VERIFIER_SHA256" \
    "$QUALITY_SIGNER_SHA256" "$QUALITY_INSTALLER_SHA256" "$CURRENT_ID" "$CURRENT_ARCHIVE_SHA256" \
    "$CURRENT_MANIFEST_SHA256" "$CURRENT_SBOM_SHA256" "$CURRENT_BUNDLE_INVENTORY_SHA256" \
    "$PREVIOUS_ID" "$PREVIOUS_ARCHIVE_SHA256" "$PREVIOUS_MANIFEST_SHA256" "$PREVIOUS_SBOM_SHA256" \
    "$P4_TEST_FINGERPRINT" "$STATE_SENTINEL_SHA256" <<'PY'
import json, pathlib, sys
(
    out,p3_path,activation_path,blockers_path,head,parent,checkout_sha,metadata_sha,
    started,finished,p3_sha,gate4_sha,builder_sha,verifier_sha,signer_sha,installer_sha,
    current_id,current_archive_sha,current_manifest_sha,current_sbom_sha,current_inventory_sha,
    previous_id,previous_archive_sha,previous_manifest_sha,previous_sbom_sha,test_fingerprint,state_sha,
)=sys.argv[1:]
p3=json.load(open(p3_path,encoding='utf-8'))
activation=json.load(open(activation_path,encoding='utf-8'))
blockers=[line for line in pathlib.Path(blockers_path).read_text().splitlines() if line]
receipt={
    'schema':'cerebro.tidex.release_readiness_receipt/v1',
    'gate':'P4',
    'result':'technical-passed',
    'technical_release_ready':True,
    'public_promotion_ready':False,
    'public_promotion_blockers':blockers,
    'head_commit':head,
    'previous_commit':parent,
    'checkout_manifest_sha256':checkout_sha,
    'checkout_metadata_sha256':metadata_sha,
    'started_epoch':int(started),
    'finished_epoch':int(finished),
    'duration_seconds':int(finished)-int(started),
    'cumulative_p3':{
        'result':p3.get('result'),
        'receipt_sha256':p3_sha,
        'p2_coverage':p3.get('p2_coverage'),
        'required_assurance_tests':len(p3.get('required_assurance_tests',[])),
    },
    'release_tooling_sha256':{
        'gate4':gate4_sha,
        'builder':builder_sha,
        'verifier':verifier_sha,
        'signer':signer_sha,
        'install_manager':installer_sha,
    },
    'current_release':{
        'release_id':current_id,
        'archive_sha256':current_archive_sha,
        'manifest_sha256':current_manifest_sha,
        'sbom_sha256':current_sbom_sha,
        'reproducible_container_inventory_sha256':current_inventory_sha,
        'complete_bundle_reproductions':2,
        'binary_builds_per_bundle':2,
    },
    'previous_release':{
        'release_id':previous_id,
        'archive_sha256':previous_archive_sha,
        'manifest_sha256':previous_manifest_sha,
        'sbom_sha256':previous_sbom_sha,
        'source':'independent local clone of previous_commit',
    },
    'openpgp_test':{
        'fingerprint':test_fingerprint,
        'test_only':True,
        'subjects':['SHA256SUMS','release-manifest.json','release.tar.zst','release.tar.zst.sha256'],
        'production_identity_created':False,
    },
    'installation_lifecycle':{
        'upgrade':True,
        'derived_pointer_recovery_without_revision_advance':True,
        'installed_runtime_empty_state_fail_closed':True,
        'rollback':True,
        'roll_forward':True,
        'inactive_uninstall':True,
        'active_uninstall_rejected':True,
        'rollback_without_previous_rejected':True,
        'private_state_sha256':state_sha,
        'final_activation':activation,
    },
    'limitations':[
        'OpenPGP evidence uses an ephemeral test key, not a production distribution identity',
        'public promotion remains separate from technical release readiness',
        'product licensing is not inferred or invented by the gate',
        'target portability is demonstrated for the host target exercised by this gate',
    ],
}
path=pathlib.Path(out)
path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
PY
sha256sum "$QUALITY_P4_RECEIPT_PATH" > "${QUALITY_P4_RECEIPT_PATH}.sha256"

printf 'P4 TÉCNICA SUPERADA: P3 acumulativa, bundle reproducible, SPDX, firma OpenPGP de prueba, instalación, upgrade, rollback y uninstall pasan sin modificar el checkout. receipt=%s\n' "$QUALITY_P4_RECEIPT_PATH"
printf 'P4 PROMOCIÓN PÚBLICA NO LISTA: %s\n' "$(paste -sd, "$PUBLIC_BLOCKERS_FILE")"
