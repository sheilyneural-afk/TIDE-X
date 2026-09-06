#!/usr/bin/env bash
set -euo pipefail

umask 077
QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
VERIFIER="$QUALITY_ROOT/quality/verify-release.sh"

fail(){ printf 'release installation rejected: %s\n' "$1" >&2; exit 2; }
usage(){
    cat >&2 <<'USAGE'
usage:
  quality/manage-release-installation.sh install <release-dir> <release-archive> <install-root> <state-root>
  quality/manage-release-installation.sh activate <install-root> <release-id>
  quality/manage-release-installation.sh rollback <install-root>
  quality/manage-release-installation.sh uninstall <install-root> <release-id>
  quality/manage-release-installation.sh verify <install-root>
USAGE
}

[[ -f "$VERIFIER" && ! -L "$VERIFIER" && -x "$VERIFIER" ]] || fail 'release_verifier_unavailable'
for tool in python3 flock cp mv rm diff uname find install mktemp rmdir; do command -v "$tool" >/dev/null 2>&1 || fail "required_tool_missing:$tool"; done

normalize_dir(){
    local path=$1 mode=$2 create=$3
    python3 - "$path" "$mode" "$create" <<'PY'
import os, stat, sys
from pathlib import Path
raw,mode_text,create=sys.argv[1:]
p=Path(raw)
if not p.is_absolute():
    print('release installation rejected: path_must_be_absolute',file=sys.stderr); raise SystemExit(2)
parent=p.parent
if not parent.exists():
    print('release installation rejected: parent_directory_missing',file=sys.stderr); raise SystemExit(2)
current=Path(p.parts[0])
for part in p.parts[1:-1]:
    current/=part
    st=os.lstat(current)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        print(f'release installation rejected: path_component_invalid:{current}',file=sys.stderr); raise SystemExit(2)
if p.exists() or os.path.lexists(p):
    st=os.lstat(p)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        print('release installation rejected: directory_invalid',file=sys.stderr); raise SystemExit(2)
elif create=='1':
    requested=int(mode_text,8)
    os.mkdir(p,requested)
    os.chmod(p,requested)
else:
    print('release installation rejected: directory_missing',file=sys.stderr); raise SystemExit(2)
print(os.path.realpath(p))
PY
}

host_target(){
    local os arch
    os=$(uname -s); arch=$(uname -m)
    case "$os:$arch" in
        Linux:x86_64) echo x86_64-unknown-linux-gnu ;;
        Linux:aarch64|Linux:arm64) echo aarch64-unknown-linux-gnu ;;
        *) fail "unsupported_host_target:${os}:${arch}" ;;
    esac
}

read_manifest_field(){
    local manifest=$1 field=$2
    python3 - "$manifest" "$field" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
v=x
for part in sys.argv[2].split('.'):
    v=v[part]
if not isinstance(v,(str,int,bool)):
    raise SystemExit(2)
print(str(v).lower() if isinstance(v,bool) else v)
PY
}

ensure_state_root_private(){
    local root=$1
    python3 - "$root" <<'PY'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_absolute(): raise SystemExit(2)
current=Path(p.parts[0])
for part in p.parts[1:]:
    current/=part
    st=os.lstat(current)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        print(f'release installation rejected: state_root_path_invalid:{current}',file=sys.stderr); raise SystemExit(2)
mode=stat.S_IMODE(os.lstat(p).st_mode)
if mode & 0o077:
    print(f'release installation rejected: state_root_permissions_too_open:{mode:o}',file=sys.stderr); raise SystemExit(2)
PY
}

ensure_install_root_safe(){
    local root=$1
    python3 - "$root" <<'PY'
import os,stat,sys
from pathlib import Path
root=Path(sys.argv[1])
st=os.lstat(root)
if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
    print('release installation rejected: install_root_invalid',file=sys.stderr); raise SystemExit(2)
mode=stat.S_IMODE(st.st_mode)
if mode & 0o022:
    print(f'release installation rejected: install_root_writable_by_others:{mode:o}',file=sys.stderr); raise SystemExit(2)
releases=root/'releases'
if os.path.lexists(releases):
    rst=os.lstat(releases)
    if stat.S_ISLNK(rst.st_mode) or not stat.S_ISDIR(rst.st_mode):
        print('release installation rejected: releases_root_invalid',file=sys.stderr); raise SystemExit(2)
    rmode=stat.S_IMODE(rst.st_mode)
    if rmode & 0o022:
        print(f'release installation rejected: releases_root_writable_by_others:{rmode:o}',file=sys.stderr); raise SystemExit(2)
PY
}

write_json_atomic(){
    local path=$1 payload=$2
    python3 - "$path" "$payload" <<'PY'
import json,os,stat,sys,tempfile
path,payload=sys.argv[1:]
parent=os.path.dirname(path)
if os.path.lexists(path):
    st=os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        print('release installation rejected: authority_file_invalid',file=sys.stderr); raise SystemExit(2)
data=json.dumps(json.loads(payload),sort_keys=True,separators=(',',':')).encode()+b'\n'
fd,tmp=tempfile.mkstemp(prefix='.authority.',dir=parent)
try:
    os.fchmod(fd,0o644)
    os.write(fd,data); os.fsync(fd); os.close(fd); fd=-1
    os.replace(tmp,path)
    dfd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(dfd)
    finally: os.close(dfd)
finally:
    if fd>=0: os.close(fd)
    try: os.unlink(tmp)
    except FileNotFoundError: pass
PY
}

load_installation(){
    local root=$1
    python3 - "$root/installation.json" <<'PY'
import json,os,stat,sys
p=sys.argv[1]
st=os.lstat(p) if os.path.exists(p) else None
if st is None or stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)&0o022: raise SystemExit(2)
x=json.load(open(p,encoding='utf-8'))
if x.get('schema')!='cerebro.tidex.installation/v1': raise SystemExit(2)
if not isinstance(x.get('state_root'),str) or not isinstance(x.get('target'),str): raise SystemExit(2)
print(x['state_root']); print(x['target'])
PY
}

load_activation(){
    local root=$1
    python3 - "$root/activation.json" <<'PY'
import json,os,stat,sys
p=sys.argv[1]
st=os.lstat(p) if os.path.exists(p) else None
if st is None or stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)&0o022: raise SystemExit(2)
x=json.load(open(p,encoding='utf-8'))
if x.get('schema')!='cerebro.tidex.activation/v1': raise SystemExit(2)
active=x.get('active'); previous=x.get('previous'); revision=x.get('revision')
for v in (active,previous):
    if v is not None and (not isinstance(v,str) or '/' in v or v in ('','.','..')): raise SystemExit(2)
if not isinstance(revision,int) or revision<0: raise SystemExit(2)
if active is None:
    if previous is not None or revision != 0: raise SystemExit(2)
else:
    if revision < 1 or previous == active: raise SystemExit(2)
print(active if active is not None else '-')
print(previous if previous is not None else '-')
print(revision)
PY
}

read_installation_authority(){
    local raw
    raw=$(load_installation "$1") || fail 'installation_authority_invalid'
    mapfile -t INSTALLATION_AUTHORITY <<< "$raw"
    [[ ${#INSTALLATION_AUTHORITY[@]} -eq 2 ]] || fail 'installation_authority_invalid'
}

read_activation_authority(){
    local raw
    raw=$(load_activation "$1") || fail 'activation_authority_invalid'
    mapfile -t ACTIVATION_AUTHORITY <<< "$raw"
    [[ ${#ACTIVATION_AUTHORITY[@]} -eq 3 ]] || fail 'activation_authority_invalid'
}

write_activation(){
    local root=$1 active=$2 previous=$3 revision=$4
    local active_json previous_json payload
    [[ "$active" == - ]] && active_json=null || active_json=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$active")
    [[ "$previous" == - ]] && previous_json=null || previous_json=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$previous")
    payload="{\"schema\":\"cerebro.tidex.activation/v1\",\"active\":$active_json,\"previous\":$previous_json,\"revision\":$revision}"
    write_json_atomic "$root/activation.json" "$payload"
}

validate_release_id(){ [[ "$1" =~ ^cerebro-tidex-[A-Za-z0-9._+-]+-[A-Za-z0-9_.-]+-[0-9a-f]{12}$ ]] || fail 'release_id_invalid'; }

check_release_dir(){
    local root=$1
    local id=$2
    local expected_target=$3
    local dir="$root/releases/$id"
    local actual_target
    [[ -d "$dir" && ! -L "$dir" ]] || fail "installed_release_missing:$id"
    "$VERIFIER" "$dir" >/dev/null
    actual_target=$(read_manifest_field "$dir/release-manifest.json" target) || fail "installed_release_target_unreadable:$id"
    [[ "$actual_target" == "$expected_target" ]] || fail "installed_release_target_mismatch:$id"
}

sync_current_link(){
    local root=$1 active=$2
    python3 - "$root" "$active" <<'PY'
import os,stat,sys,tempfile
root,active=sys.argv[1:]
current=os.path.join(root,'current')
if active=='-':
    if os.path.lexists(current):
        st=os.lstat(current)
        if not stat.S_ISLNK(st.st_mode):
            print('release installation rejected: current_pointer_not_symlink',file=sys.stderr); raise SystemExit(2)
        os.unlink(current)
    raise SystemExit(0)
expected=f'releases/{active}'
target=os.path.join(root,'releases',active)
if not os.path.isdir(target) or stat.S_ISLNK(os.lstat(target).st_mode):
    print('release installation rejected: active_release_target_invalid',file=sys.stderr); raise SystemExit(2)
if os.path.lexists(current):
    st=os.lstat(current)
    if not stat.S_ISLNK(st.st_mode):
        print('release installation rejected: current_pointer_not_symlink',file=sys.stderr); raise SystemExit(2)
    if os.readlink(current)==expected: raise SystemExit(0)
tmp=os.path.join(root,f'.current.{os.getpid()}')
try:
    os.symlink(expected,tmp)
    os.replace(tmp,current)
    dfd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(dfd)
    finally: os.close(dfd)
finally:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
PY
}

check_current_link(){
    local root=$1 active=$2
    python3 - "$root" "$active" <<'PY'
import os,stat,sys
root,active=sys.argv[1:]; current=os.path.join(root,'current')
if active=='-':
    if os.path.lexists(current): raise SystemExit(2)
    raise SystemExit(0)
if not os.path.lexists(current) or not stat.S_ISLNK(os.lstat(current).st_mode): raise SystemExit(2)
if os.readlink(current)!=f'releases/{active}': raise SystemExit(2)
PY
}

open_install_lock(){
    local root=$1
    local lock="$root/.install.lock"
    if [[ ! -e "$lock" && ! -L "$lock" ]]; then
        ( set -o noclobber; umask 077; : > "$lock" ) 2>/dev/null || true
    fi
    python3 - "$lock" <<'PY' || exit 2
import os,stat,sys
p=sys.argv[1]
try: st=os.lstat(p)
except FileNotFoundError:
    print('release installation rejected: install_lock_missing',file=sys.stderr); raise SystemExit(2)
if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
    print('release installation rejected: install_lock_invalid',file=sys.stderr); raise SystemExit(2)
if stat.S_IMODE(st.st_mode) & 0o077:
    print('release installation rejected: install_lock_permissions_too_open',file=sys.stderr); raise SystemExit(2)
PY
    exec 9<>"$lock"
    flock -x 9
    local fd_identity path_identity
    fd_identity=$(stat -Lc '%d:%i' "/proc/$$/fd/9")
    path_identity=$(stat -Lc '%d:%i' "$lock")
    [[ "$fd_identity" == "$path_identity" ]] || fail 'install_lock_identity_changed'
}

cleanup_stale_stages(){
    local releases=$1 entry
    shopt -s nullglob
    for entry in "$releases"/.stage.*; do
        [[ -d "$entry" && ! -L "$entry" ]] || fail 'stale_stage_path_invalid'
        rm -rf --one-file-system -- "$entry"
    done
    shopt -u nullglob
}

initialize_installation(){
    local root=$1 state=$2 target=$3
    install -d -m 0755 "$root/releases"
    if [[ ! -e "$root/installation.json" ]]; then
        write_json_atomic "$root/installation.json" "{\"schema\":\"cerebro.tidex.installation/v1\",\"state_root\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$state"),\"target\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$target")}"
    fi
    read_installation_authority "$root"
    [[ "${INSTALLATION_AUTHORITY[0]}" == "$state" && "${INSTALLATION_AUTHORITY[1]}" == "$target" ]] || fail 'installation_authority_mismatch'
    if [[ ! -e "$root/activation.json" ]]; then write_activation "$root" - - 0; fi
    read_activation_authority "$root"
}

command=${1:-}
case "$command" in
install)
    [[ $# -eq 5 ]] || { usage; exit 2; }
    release_input=$2; archive_input=$3; install_input=$4; state_input=$5
    "$VERIFIER" "$release_input" "$archive_input" >/dev/null
    release_dir=$(cd -- "$release_input" && pwd -P)
    release_id=$(read_manifest_field "$release_dir/release-manifest.json" release_id)
    release_target=$(read_manifest_field "$release_dir/release-manifest.json" target)
    validate_release_id "$release_id"
    [[ "$release_target" == "$(host_target)" ]] || fail 'release_target_does_not_match_host'
    install_root=$(normalize_dir "$install_input" 0755 1) || exit 2
    ensure_install_root_safe "$install_root" || exit 2
    state_root=$(normalize_dir "$state_input" 0700 1) || exit 2
    ensure_state_root_private "$state_root" || exit 2
    case "$state_root/" in "$install_root/"*) fail 'state_root_inside_install_root' ;; esac
    case "$install_root/" in "$state_root/"*) fail 'install_root_inside_state_root' ;; esac
    open_install_lock "$install_root"
    initialize_installation "$install_root" "$state_root" "$release_target"
    ensure_install_root_safe "$install_root" || exit 2
    cleanup_stale_stages "$install_root/releases"
    final="$install_root/releases/$release_id"
    if [[ -e "$final" ]]; then
        [[ -d "$final" && ! -L "$final" ]] || fail 'installed_release_path_invalid'
        "$VERIFIER" "$final" >/dev/null
        diff -qr --no-dereference "$release_dir" "$final" >/dev/null || fail 'installed_release_conflicts_with_existing_identity'
        printf 'INSTALLED release_id=%s already=true install_root=%s state_root=%s\n' "$release_id" "$install_root" "$state_root"
        exit 0
    fi
    stage_parent=$(mktemp -d "$install_root/releases/.stage.XXXXXX")
    cleanup_stage(){ rm -rf -- "$stage_parent"; }
    trap cleanup_stage EXIT
    cp -a -- "$release_dir" "$stage_parent/$release_id"
    "$VERIFIER" "$stage_parent/$release_id" >/dev/null
    mv -T -- "$stage_parent/$release_id" "$final"
    rmdir "$stage_parent"
    trap - EXIT
    printf 'INSTALLED release_id=%s already=false install_root=%s state_root=%s\n' "$release_id" "$install_root" "$state_root"
    ;;
activate)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    install_root=$(normalize_dir "$2" 0755 0) || exit 2; release_id=$3; validate_release_id "$release_id"
    ensure_install_root_safe "$install_root" || exit 2
    open_install_lock "$install_root"
    read_installation_authority "$install_root"
    ensure_state_root_private "${INSTALLATION_AUTHORITY[0]}" || exit 2
    read_activation_authority "$install_root"
    sync_current_link "$install_root" "${ACTIVATION_AUTHORITY[0]}"
    check_release_dir "$install_root" "$release_id" "${INSTALLATION_AUTHORITY[1]}"
    if [[ "${ACTIVATION_AUTHORITY[0]}" == "$release_id" ]]; then
        printf 'ACTIVATED release_id=%s revision=%s already=true\n' "$release_id" "${ACTIVATION_AUTHORITY[2]}"; exit 0
    fi
    new_revision=$((ACTIVATION_AUTHORITY[2]+1)); write_activation "$install_root" "$release_id" "${ACTIVATION_AUTHORITY[0]}" "$new_revision"
    sync_current_link "$install_root" "$release_id"
    printf 'ACTIVATED release_id=%s revision=%s already=false previous=%s\n' "$release_id" "$new_revision" "${ACTIVATION_AUTHORITY[0]}"
    ;;
rollback)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    install_root=$(normalize_dir "$2" 0755 0) || exit 2
    ensure_install_root_safe "$install_root" || exit 2
    open_install_lock "$install_root"
    read_installation_authority "$install_root"
    ensure_state_root_private "${INSTALLATION_AUTHORITY[0]}" || exit 2
    read_activation_authority "$install_root"
    sync_current_link "$install_root" "${ACTIVATION_AUTHORITY[0]}"
    [[ "${ACTIVATION_AUTHORITY[1]}" != - ]] || fail 'rollback_target_missing'
    check_release_dir "$install_root" "${ACTIVATION_AUTHORITY[1]}" "${INSTALLATION_AUTHORITY[1]}"
    new_revision=$((ACTIVATION_AUTHORITY[2]+1)); old_active=${ACTIVATION_AUTHORITY[0]}; new_active=${ACTIVATION_AUTHORITY[1]}
    write_activation "$install_root" "$new_active" "$old_active" "$new_revision"
    sync_current_link "$install_root" "$new_active"
    printf 'ROLLED_BACK active=%s previous=%s revision=%s\n' "$new_active" "$old_active" "$new_revision"
    ;;
uninstall)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    install_root=$(normalize_dir "$2" 0755 0) || exit 2; release_id=$3; validate_release_id "$release_id"
    ensure_install_root_safe "$install_root" || exit 2
    open_install_lock "$install_root"
    read_installation_authority "$install_root"
    state_root=${INSTALLATION_AUTHORITY[0]}; ensure_state_root_private "$state_root" || exit 2
    read_activation_authority "$install_root"
    sync_current_link "$install_root" "${ACTIVATION_AUTHORITY[0]}"
    [[ "$release_id" != "${ACTIVATION_AUTHORITY[0]}" ]] || fail 'cannot_uninstall_active_release'
    target="$install_root/releases/$release_id"
    [[ -d "$target" && ! -L "$target" ]] || fail 'installed_release_missing'
    if [[ "$release_id" == "${ACTIVATION_AUTHORITY[1]}" ]]; then
        new_revision=$((ACTIVATION_AUTHORITY[2]+1)); write_activation "$install_root" "${ACTIVATION_AUTHORITY[0]}" - "$new_revision"
    fi
    rm -rf --one-file-system -- "$target"
    [[ -d "$state_root" ]] || fail 'state_root_lost_during_uninstall'
    printf 'UNINSTALLED release_id=%s state_preserved=%s\n' "$release_id" "$state_root"
    ;;
verify)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    install_root=$(normalize_dir "$2" 0755 0) || exit 2
    ensure_install_root_safe "$install_root" || exit 2
    read_installation_authority "$install_root"
    state_root=${INSTALLATION_AUTHORITY[0]}; target=${INSTALLATION_AUTHORITY[1]}; ensure_state_root_private "$state_root" || exit 2
    [[ "$target" == "$(host_target)" ]] || fail 'installation_target_does_not_match_host'
    read_activation_authority "$install_root"
    check_current_link "$install_root" "${ACTIVATION_AUTHORITY[0]}" || fail 'current_pointer_mismatch'
    bad_entry=$(find "$install_root/releases" -mindepth 1 -maxdepth 1 ! -type d -print -quit)
    [[ -z "$bad_entry" ]] || fail "release_store_unexpected_entry:$(basename "$bad_entry")"
    count=0
    while IFS= read -r -d '' dir; do
        id=$(basename "$dir"); validate_release_id "$id"; check_release_dir "$install_root" "$id" "$target"; count=$((count+1))
    done < <(find "$install_root/releases" -mindepth 1 -maxdepth 1 -type d -print0)
    [[ "${ACTIVATION_AUTHORITY[0]}" == - || -d "$install_root/releases/${ACTIVATION_AUTHORITY[0]}" ]] || fail 'active_release_missing'
    [[ "${ACTIVATION_AUTHORITY[1]}" == - || -d "$install_root/releases/${ACTIVATION_AUTHORITY[1]}" ]] || fail 'previous_release_missing'
    printf 'INSTALLATION_VERIFIED active=%s previous=%s revision=%s releases=%s state_root=%s\n' "${ACTIVATION_AUTHORITY[0]}" "${ACTIVATION_AUTHORITY[1]}" "${ACTIVATION_AUTHORITY[2]}" "$count" "$state_root"
    ;;
*) usage; exit 2 ;;
esac
