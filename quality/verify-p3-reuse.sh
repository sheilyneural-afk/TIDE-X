#!/usr/bin/env bash
set -euo pipefail
umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
EVIDENCE_DIR="$QUALITY_ROOT/quality/evidence/p3"
RECEIPT="$EVIDENCE_DIR/receipt.json"
ANCHOR="$EVIDENCE_DIR/reuse-anchor.json"
SIDE="$EVIDENCE_DIR/receipt.sha256"

fail(){ printf 'P3 reuse rejected: %s\n' "$1" >&2; exit 2; }
for tool in git python3 sha256sum rustc; do command -v "$tool" >/dev/null 2>&1 || fail "required_tool_missing:$tool"; done
[[ -f "$RECEIPT" && ! -L "$RECEIPT" ]] || fail 'canonical_receipt_missing'
[[ -f "$ANCHOR" && ! -L "$ANCHOR" ]] || fail 'reuse_anchor_missing'
[[ -f "$SIDE" && ! -L "$SIDE" ]] || fail 'receipt_checksum_missing'
cd "$QUALITY_ROOT"
[[ -z "$(git status --porcelain=v1)" ]] || fail 'working_tree_not_clean'
(
  cd "$EVIDENCE_DIR"
  sha256sum --strict -c receipt.sha256 >/dev/null
) || fail 'canonical_receipt_checksum_invalid'

OUTPUT=${QUALITY_P3_REUSE_RECEIPT_PATH:-}
if [[ -n "$OUTPUT" ]]; then
    [[ "$OUTPUT" == /* ]] || OUTPUT="$QUALITY_ROOT/$OUTPUT"
    case "$OUTPUT" in "$QUALITY_ROOT"|"$QUALITY_ROOT"/*) fail 'reuse_receipt_must_be_outside_checkout' ;; esac
    mkdir -p "$(dirname "$OUTPUT")"
fi

python3 - "$RECEIPT" "$ANCHOR" "$OUTPUT" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

receipt_path, anchor_path, output_path = sys.argv[1:]
root = pathlib.Path.cwd()
receipt_bytes = pathlib.Path(receipt_path).read_bytes()
receipt = json.loads(receipt_bytes)
anchor = json.load(open(anchor_path, encoding='utf-8'))

if anchor.get('schema') != 'cerebro.tidex.p3_reuse_anchor/v1':
    raise SystemExit('P3 reuse rejected: anchor_schema_invalid')
if receipt.get('schema') != 'cerebro.tidex.quality_assurance_receipt/v1' or receipt.get('gate') != 'P3' or receipt.get('result') != 'passed':
    raise SystemExit('P3 reuse rejected: source_receipt_invalid')
receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
if receipt_sha != anchor.get('receipt_sha256'):
    raise SystemExit('P3 reuse rejected: receipt_anchor_digest_mismatch')

snapshot = anchor.get('frozen_snapshot_commit')
parent = anchor.get('frozen_snapshot_parent')
tree = anchor.get('frozen_snapshot_tree')
if not all(isinstance(x, str) and len(x) == 40 for x in (snapshot, parent, tree)):
    raise SystemExit('P3 reuse rejected: frozen_git_identity_invalid')
actual_parent = subprocess.check_output(['git','rev-parse',snapshot+'^'], text=True).strip()
actual_tree = subprocess.check_output(['git','rev-parse',snapshot+'^{tree}'], text=True).strip()
if actual_parent != parent or actual_tree != tree or receipt.get('head_commit') != parent:
    raise SystemExit('P3 reuse rejected: frozen_git_binding_mismatch')
if subprocess.run(['git','merge-base','--is-ancestor',snapshot,'HEAD']).returncode != 0:
    raise SystemExit('P3 reuse rejected: frozen_snapshot_not_ancestor')

# Reconstruct the exact historical full-checkout content manifest. Git provides
# tracked bytes; the anchor supplies only hashes+paths for ignored files that
# were included by Gate3's original `find` snapshot.
raw = subprocess.check_output(['git','ls-tree','-r','-z',snapshot])
full = {}
for entry in raw.split(b'\0'):
    if not entry:
        continue
    meta, path_b = entry.split(b'\t', 1)
    mode, typ, obj = meta.split()
    if mode not in (b'100644', b'100755'):
        continue
    path = path_b.decode()
    data = subprocess.check_output(['git','cat-file','blob',obj.decode()])
    full[path] = hashlib.sha256(data).hexdigest()
for item in anchor.get('ignored_snapshot_files', []):
    path = item.get('path')
    digest = item.get('sha256')
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64 or path in full:
        raise SystemExit('P3 reuse rejected: ignored_snapshot_record_invalid')
    full[path] = digest
stream = b''.join(full[path].encode() + b'  ./' + path.encode() + b'\0' for path in sorted(full))
historical_manifest = hashlib.sha256(stream).hexdigest()
if historical_manifest != receipt.get('checkout_manifest_sha256') or historical_manifest != anchor.get('receipt_checkout_manifest_sha256'):
    raise SystemExit('P3 reuse rejected: historical_checkout_manifest_mismatch')

ROOT_FILES = {
    'Cargo.toml','Cargo.lock','build.rs','rust-toolchain.toml','deny.toml','.cargo/config.toml',
    'fuzz/Cargo.toml','fuzz/Cargo.lock','fuzz/deny.toml',
    'quality/gate0-release.sh','quality/gate0-empty-state.sh','quality/gate1-tooling.sh',
    'quality/gate2-verification.sh','quality/gate3-assurance.sh',
}
PREFIXES = ('src/','tests/','fuzz/fuzz_targets/','fuzz/seeds/')
DOMAIN = b'CEREBRO:TIDEX:P3-REUSABLE-INPUTS:v1\0'

def selected_snapshot_rows():
    rows=[]
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        meta,path_b=entry.split(b'\t',1)
        mode,typ,obj=meta.split(); path=path_b.decode()
        if path not in ROOT_FILES and not path.startswith(PREFIXES):
            continue
        if mode not in (b'100644',b'100755'):
            raise SystemExit('P3 reuse rejected: selected_snapshot_mode_invalid')
        data=subprocess.check_output(['git','cat-file','blob',obj.decode()])
        rows.append((path,data,1 if mode==b'100755' else 0))
    rows.sort(key=lambda x:x[0])
    return rows

def selected_current_rows(expected_paths):
    selected=[]
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        rel=path.relative_to(root).as_posix()
        if rel in ROOT_FILES or rel.startswith(PREFIXES):
            selected.append(rel)
    selected=sorted(selected)
    if selected != expected_paths:
        raise SystemExit('P3 reuse rejected: reusable_input_path_set_changed')
    rows=[]
    for rel in selected:
        path=root/rel
        mode=os.stat(path).st_mode
        rows.append((rel,path.read_bytes(),1 if mode & 0o111 else 0))
    return rows

def digest_rows(rows):
    h=hashlib.sha256(); h.update(DOMAIN)
    for path,data,executable in rows:
        for field in (path.encode(),bytes([executable]),data):
            h.update(len(field).to_bytes(8,'big')); h.update(field)
    return h.hexdigest()

snapshot_rows=selected_snapshot_rows()
expected_paths=[r[0] for r in snapshot_rows]
current_rows=selected_current_rows(expected_paths)
snapshot_digest=digest_rows(snapshot_rows)
current_digest=digest_rows(current_rows)
expected_digest=anchor.get('reusable_inputs_sha256')
expected_count=anchor.get('reusable_input_file_count')
if len(snapshot_rows) != expected_count or snapshot_digest != expected_digest:
    raise SystemExit('P3 reuse rejected: anchor_reusable_input_binding_invalid')
if current_digest != expected_digest or len(current_rows) != expected_count:
    raise SystemExit('P3 reuse rejected: current_reusable_inputs_changed')

for key,path in (('gate2_script_sha256','quality/gate2-verification.sh'),('gate3_script_sha256','quality/gate3-assurance.sh')):
    actual=hashlib.sha256((root/path).read_bytes()).hexdigest()
    if actual != anchor.get(key):
        raise SystemExit(f'P3 reuse rejected: {key}_changed')
receipt_evidence=receipt.get('evidence_sha256',{})
if receipt_evidence.get('gate2_script') != anchor.get('gate2_script_sha256') or receipt_evidence.get('gate3_script') != anchor.get('gate3_script_sha256'):
    raise SystemExit('P3 reuse rejected: receipt_gate_script_binding_invalid')
if len(receipt.get('required_assurance_tests',[])) != anchor.get('required_assurance_tests'):
    raise SystemExit('P3 reuse rejected: assurance_test_count_mismatch')
if receipt.get('p2_coverage') != anchor.get('p2_coverage'):
    raise SystemExit('P3 reuse rejected: coverage_anchor_mismatch')

stable=subprocess.check_output(['rustc','-Vv'],text=True).strip()
nightly=subprocess.check_output(['rustc','+nightly','-Vv'],text=True).strip()
if stable != receipt.get('toolchains',{}).get('stable_rustc_vv','').strip():
    raise SystemExit('P3 reuse rejected: stable_toolchain_drift')
if nightly != receipt.get('toolchains',{}).get('nightly_rustc_vv','').strip():
    raise SystemExit('P3 reuse rejected: nightly_toolchain_drift')

result={
    'schema':'cerebro.tidex.p3_reuse_receipt/v1',
    'result':'reused',
    'current_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'source_p3_receipt_sha256':receipt_sha,
    'frozen_snapshot_commit':snapshot,
    'historical_checkout_manifest_sha256':historical_manifest,
    'reusable_inputs_sha256':current_digest,
    'reusable_input_file_count':len(current_rows),
    'required_assurance_tests':len(receipt.get('required_assurance_tests',[])),
    'p2_coverage':receipt.get('p2_coverage'),
    'toolchains_match':True,
    'verified_epoch':int(time.time()),
}
if output_path:
    out=pathlib.Path(output_path)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    pathlib.Path(str(out)+'.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
print('P3_REUSE_VERIFIED current_head='+result['current_head']+' inputs='+current_digest+' files='+str(len(current_rows)))
PY
