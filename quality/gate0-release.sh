#!/usr/bin/env bash
set -uo pipefail

GATE0_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GATE0_TMP=$(mktemp -d /tmp/tidex-gate0-release.XXXXXX)
GATE0_TARGET="$GATE0_TMP/target"
GATE0_DENY_CARGO_HOME="$GATE0_TMP/cargo-home-deny"
GATE0_FAILURES=0
GATE0_BLOCKED=0

cleanup() {
    case "$GATE0_TMP" in
        /tmp/tidex-gate0-release.*) rm -rf -- "$GATE0_TMP" ;;
        *)
            echo "P0 ERROR: ruta temporal inesperada; se rechaza eliminarla: $GATE0_TMP" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

export CARGO_TARGET_DIR="$GATE0_TARGET"
export CARGO_NET_OFFLINE=true
cd "$GATE0_ROOT" || exit 1

pass() {
    printf 'P0 OK: %s\n' "$1"
}

fail() {
    printf 'P0 FALLO: %s\n' "$1" >&2
    GATE0_FAILURES=$((GATE0_FAILURES + 1))
}

run_check() {
    local description=$1
    shift
    if "$@"; then
        pass "$description"
    else
        fail "$description"
    fi
}

check_toolchain() {
    local expected actual
    expected=$(sed -n 's/^channel = "\([^"]*\)"$/\1/p' rust-toolchain.toml)
    actual=$(rustc -Vv | sed -n 's/^release: //p')
    [[ -n "$expected" && "$actual" == "$expected" ]] || {
        printf 'toolchain esperado=%s actual=%s\n' "$expected" "$actual" >&2
        return 1
    }
}

check_no_residue() {
    local residue_report="$GATE0_TMP/residue.txt"
    local relative
    : > "$residue_report"

    for relative in \
        target \
        fuzz/target \
        fuzz/corpus \
        fuzz/artifacts \
        models \
        state \
        artifacts \
        evaluations
    do
        [[ ! -e "$relative" ]] || printf '%s\n' "$relative" >> "$residue_report"
    done

    find . \
        -path ./.git -prune -o \
        \( -path ./fuzz/corpus -o -path ./fuzz/artifacts \) -print -prune -o \
        -type d \( \
            -name target -o \
            -name .venv -o \
            -name node_modules -o \
            -name __pycache__ -o \
            -name .pytest_cache -o \
            -name .mypy_cache -o \
            -name .ruff_cache \
        \) -print -prune -o \
        -type f \( \
            -name '*.pyc' -o \
            -name '*.pyo' -o \
            -name '*.profraw' -o \
            -name 'fuzz-*.log' \
        \) -print >> "$residue_report"

    sort -u -o "$residue_report" "$residue_report"
    if [[ -s "$residue_report" ]]; then
        echo 'residuos detectados:' >&2
        sed 's/^/  - /' "$residue_report" >&2
        return 1
    fi
}

check_no_checkout_stores() {
    local store report="$GATE0_TMP/checkout-stores.txt"
    : > "$report"

    for store in models state artifacts evaluations; do
        [[ ! -e "$store" ]] || printf '%s\n' "$store" >> "$report"
    done

    if [[ -s "$report" ]]; then
        echo 'almacenes de ejecución presentes dentro del checkout:' >&2
        sed 's/^/  - /' "$report" >&2
        return 1
    fi
}

check_metadata() {
    local manifest=$1
    local output=$2
    cargo metadata --manifest-path "$manifest" --offline --locked \
        --format-version 1 --no-deps > "$output"
    [[ -s "$output" ]]
}

check_default_target_outside_checkout() {
    local manifest=$1
    local metadata="$GATE0_TMP/default-target-${manifest//[\/.]/_}-metadata.json"
    local target_dir
    env -u CARGO_TARGET_DIR cargo metadata --manifest-path "$manifest" --offline --locked \
        --format-version 1 --no-deps > "$metadata" || return 1
    target_dir=$(sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' "$metadata")
    [[ -n "$target_dir" ]] || return 1
    case "$target_dir/" in
        "$GATE0_ROOT/"*)
            printf 'Cargo target_directory de %s cae dentro del checkout: %s\n' \
                "$manifest" "$target_dir" >&2
            return 1
            ;;
    esac
}

prepare_cargo_deny_home() {
    local cargo_executable cargo_install_root source_cargo_home

    cargo_executable=$(command -v cargo) || return 1
    cargo_install_root=$(cd "$(dirname "$cargo_executable")/.." && pwd) || return 1
    source_cargo_home=${CARGO_HOME:-$cargo_install_root}

    [[ -d "$source_cargo_home/registry" ]] || {
        echo "no existe la caché offline de Cargo: $source_cargo_home/registry" >&2
        return 1
    }
    [[ -d "$source_cargo_home/advisory-dbs" ]] || {
        echo "no existe la base local de avisos de cargo-deny: $source_cargo_home/advisory-dbs" >&2
        return 1
    }

    install -d -m 700 "$GATE0_DENY_CARGO_HOME"
    ln -s -- "$source_cargo_home/registry" "$GATE0_DENY_CARGO_HOME/registry"
    if [[ -d "$source_cargo_home/git" ]]; then
        ln -s -- "$source_cargo_home/git" "$GATE0_DENY_CARGO_HOME/git"
    fi
    cp -a -- "$source_cargo_home/advisory-dbs" "$GATE0_DENY_CARGO_HOME/advisory-dbs"
    chmod -R u+rwX,go-rwx "$GATE0_DENY_CARGO_HOME/advisory-dbs"
}

check_empty_state_boot() {
    local probe="$GATE0_ROOT/quality/gate0-empty-state.sh"
    local empty_root="$GATE0_TMP/empty-state"

    if [[ ! -f "$probe" || -L "$probe" || ! -x "$probe" ]]; then
        echo 'P0 BLOQUEADA: no existe quality/gate0-empty-state.sh.' >&2
        echo 'Falta una prueba integral que arranque TIDE-X contra un almacén privado' >&2
        echo 'realmente vacío y demuestre rechazo fail-closed sin mutación implícita.' >&2
        echo 'No se sustituye esa evidencia por mocks ni se toca el estado residente.' >&2
        return 2
    fi

    install -d -m 700 "$empty_root"
    (
        cd "$empty_root" || exit 1
        "$probe" "$empty_root"
    )
}

run_check 'toolchain estable exactamente fijado' check_toolchain
run_check 'ausencia de residuos conocidos' check_no_residue
run_check 'ningún almacén models/state/artifacts/evaluations dentro del checkout' check_no_checkout_stores
run_check 'Cargo raíz por defecto compila fuera del checkout' \
    check_default_target_outside_checkout Cargo.toml
run_check 'Cargo fuzz por defecto compila fuera del checkout' \
    check_default_target_outside_checkout fuzz/Cargo.toml
run_check 'formato' cargo fmt --all -- --check
run_check 'metadatos raíz offline y bloqueados' \
    check_metadata Cargo.toml "$GATE0_TMP/root-metadata.json"
run_check 'metadatos fuzz offline y bloqueados' \
    check_metadata fuzz/Cargo.toml "$GATE0_TMP/fuzz-metadata.json"
run_check 'Clippy de todos los objetivos sin advertencias' \
    cargo clippy --all-targets --offline --locked -- -D warnings
run_check 'pruebas de todos los objetivos' \
    cargo test --all-targets --offline --locked
run_check 'binario primario para prueba aislada de arranque' \
    cargo build --bin cerebro-tidex --offline --locked
run_check 'auditoría de vulnerabilidades raíz sin actualizar la base local' \
    cargo audit --no-fetch --deny warnings
run_check 'auditoría de vulnerabilidades fuzz sin actualizar la base local' \
    cargo audit --file fuzz/Cargo.lock --no-fetch --deny warnings
run_check 'copia privada temporal de la base de avisos para cargo-deny' \
    prepare_cargo_deny_home
run_check 'política de dependencias raíz' \
    env CARGO_HOME="$GATE0_DENY_CARGO_HOME" cargo deny \
        --manifest-path Cargo.toml --config deny.toml \
        --offline --locked check advisories bans licenses sources
run_check 'política de dependencias fuzz' \
    env CARGO_HOME="$GATE0_DENY_CARGO_HOME" cargo deny \
        --manifest-path fuzz/Cargo.toml --config fuzz/deny.toml \
        --offline --locked check advisories bans licenses sources

if check_empty_state_boot; then
    pass 'arranque desde estado vacío aislado falla cerrado sin mutación'
else
    empty_state_status=$?
    if [[ "$empty_state_status" -eq 2 ]]; then
        GATE0_BLOCKED=1
    else
        fail 'arranque desde estado vacío aislado'
    fi
fi

if [[ "$GATE0_FAILURES" -ne 0 ]]; then
    printf 'P0 RECHAZADA: %d comprobación(es) fallaron.\n' "$GATE0_FAILURES" >&2
    exit 1
fi
if [[ "$GATE0_BLOCKED" -ne 0 ]]; then
    echo 'P0 BLOQUEADA: las comprobaciones disponibles pasan, pero falta evidencia integral de arranque vacío.' >&2
    exit 2
fi

echo 'P0 SUPERADA: todas las comprobaciones, incluido el arranque vacío real, pasan.'
