#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${1:?"uso: gate0-empty-state.sh <directorio-de-trabajo-privado>"}
PROBE_BINARY=${CARGO_TARGET_DIR:?"CARGO_TARGET_DIR no definido"}/debug/cerebro-tidex

cleanup() {
    case "$WORK_ROOT" in
        /tmp/*) find "$WORK_ROOT" -depth -delete ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT

[[ -d "$WORK_ROOT" && ! -L "$WORK_ROOT" ]]
[[ "$(stat -c '%a' "$WORK_ROOT")" == 700 ]]
[[ -x "$PROBE_BINARY" ]]

# A relocated executable and its private state are intentionally independent.
# Test two actual installations, each with an independent configured root;
# no namespace, bind mount, source checkout path, or host-specific location is
# involved in the assertion.
for installation in installation-a installation-b; do
    install_dir="$WORK_ROOT/$installation"
    state_dir="$install_dir/private-state"
    install -d -m 700 "$install_dir" "$state_dir"
    install -m 700 "$PROBE_BINARY" "$install_dir/cerebro-tidex"

    set +e
    (
        cd "$install_dir"
        TIDEX_PRIVATE_ROOT="$state_dir" ./cerebro-tidex status \
            >stdout 2>stderr
    )
    code=$?
    set -e
    printf '%s\n' "$code" >"$install_dir/exit-code"

    [[ "$(cat "$install_dir/exit-code")" == 2 ]]
    grep -Fxq 'integrity:active_skill_bank_missing' "$install_dir/stderr"
    [[ ! -s "$install_dir/stdout" ]]
    [[ -z "$(find "$state_dir" -mindepth 1 -print -quit)" ]]
done

cmp "$WORK_ROOT/installation-a/stderr" "$WORK_ROOT/installation-b/stderr"
