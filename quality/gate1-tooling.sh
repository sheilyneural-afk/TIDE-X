#!/usr/bin/env bash
set -euo pipefail

umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QUALITY_STABLE=$(sed -n 's/^channel = "\([^"]*\)"$/\1/p' "$QUALITY_ROOT/rust-toolchain.toml")
QUALITY_NIGHTLY=${QUALITY_NIGHTLY:-nightly}
QUALITY_EXPECTED_COMMIT=${QUALITY_EXPECTED_COMMIT:-0ed41eb4142dda2df61eb1145a312c1a9d62eb56}
QUALITY_FUZZ_RUNS=${QUALITY_FUZZ_RUNS:-100000}
QUALITY_REQUIRE_LSAN=${QUALITY_REQUIRE_LSAN:-0}
QUALITY_COVERAGE_FLOOR=74
QUALITY_FUNCTION_COVERAGE_FLOOR=70
QUALITY_REGION_COVERAGE_FLOOR=75
QUALITY_MIN_LINE_COVERAGE=${QUALITY_MIN_LINE_COVERAGE:-74}
QUALITY_MIN_FUNCTION_COVERAGE=${QUALITY_MIN_FUNCTION_COVERAGE:-70}
QUALITY_MIN_REGION_COVERAGE=${QUALITY_MIN_REGION_COVERAGE:-75}
QUALITY_TARGET=${QUALITY_TARGET:-x86_64-unknown-linux-gnu}
QUALITY_TMP=$(mktemp -d /tmp/tidex-gate1-quality.XXXXXX)

cleanup() {
    case "$QUALITY_TMP" in
        /tmp/tidex-gate1-quality.*) find "$QUALITY_TMP" -depth -delete ;;
        *)
            echo "quality cleanup refused unexpected path: $QUALITY_TMP" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

case "$QUALITY_FUZZ_RUNS" in
    ''|*[!0-9]*) echo 'QUALITY_FUZZ_RUNS must be an integer' >&2; exit 2 ;;
esac
case "$QUALITY_MIN_LINE_COVERAGE" in
    ''|*[!0-9]*) echo 'QUALITY_MIN_LINE_COVERAGE must be an integer' >&2; exit 2 ;;
esac
case "$QUALITY_MIN_FUNCTION_COVERAGE" in
    ''|*[!0-9]*) echo 'QUALITY_MIN_FUNCTION_COVERAGE must be an integer' >&2; exit 2 ;;
esac
case "$QUALITY_MIN_REGION_COVERAGE" in
    ''|*[!0-9]*) echo 'QUALITY_MIN_REGION_COVERAGE must be an integer' >&2; exit 2 ;;
esac
case "$QUALITY_REQUIRE_LSAN" in
    0|1) ;;
    *) echo 'QUALITY_REQUIRE_LSAN must be 0 or 1' >&2; exit 2 ;;
esac
if (( QUALITY_FUZZ_RUNS < 1 || QUALITY_FUZZ_RUNS > 10000000 )); then
    echo 'QUALITY_FUZZ_RUNS must be between 1 and 10000000' >&2
    exit 2
fi
if (( QUALITY_MIN_LINE_COVERAGE < QUALITY_COVERAGE_FLOOR || QUALITY_MIN_LINE_COVERAGE > 100 )); then
    echo "QUALITY_MIN_LINE_COVERAGE must be between $QUALITY_COVERAGE_FLOOR and 100" >&2
    exit 2
fi
if (( QUALITY_MIN_FUNCTION_COVERAGE < QUALITY_FUNCTION_COVERAGE_FLOOR || QUALITY_MIN_FUNCTION_COVERAGE > 100 )); then
    echo "QUALITY_MIN_FUNCTION_COVERAGE must be between $QUALITY_FUNCTION_COVERAGE_FLOOR and 100" >&2
    exit 2
fi
if (( QUALITY_MIN_REGION_COVERAGE < QUALITY_REGION_COVERAGE_FLOOR || QUALITY_MIN_REGION_COVERAGE > 100 )); then
    echo "QUALITY_MIN_REGION_COVERAGE must be between $QUALITY_REGION_COVERAGE_FLOOR and 100" >&2
    exit 2
fi

snapshot_checkout() {
    local prefix=$1
    find . -path ./.git -prune -o -type f -printf '%m\t%s\t%p\0' \
        | LC_ALL=C sort -z > "${prefix}.metadata"
    find . -path ./.git -prune -o -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum --zero > "${prefix}.sha256"
}

require_component() {
    local toolchain=$1 component=$2
    rustup component list --toolchain "$toolchain" \
        | grep -Eq "^${component}.*\\(installed\\)$" || {
            echo "required component missing: ${toolchain}/${component}" >&2
            exit 2
        }
}

probe_lsan_host() {
    local log="$QUALITY_TMP/lsan-probe.log"
    local status

    set +e
    CARGO_TARGET_DIR="$QUALITY_TMP/target-lsan" \
    RUSTFLAGS='-Zsanitizer=leak -Cforce-frame-pointers=yes' \
    LSAN_OPTIONS='exitcode=86:halt_on_error=1' \
        cargo "+$QUALITY_NIGHTLY" test -Zbuild-std --lib \
        low_rank_math::tests::rank_one_solution_is_exact_without_damping \
        --target "$QUALITY_TARGET" -- --exact --test-threads=1 \
        >"$log" 2>&1
    status=$?
    set -e

    if [[ "$status" -eq 0 ]]; then
        return 0
    fi
    if grep -Eq 'does not work under ptrace|requires ptrace_scope' "$log"; then
        return 2
    fi
    cat "$log" >&2
    return 1
}

cd "$QUALITY_ROOT"
snapshot_checkout "$QUALITY_TMP/checkout-before"

QUALITY_ACTUAL_COMMIT=$(rustc "+$QUALITY_NIGHTLY" -Vv | sed -n 's/^commit-hash: //p')
if [[ "$QUALITY_ACTUAL_COMMIT" != "$QUALITY_EXPECTED_COMMIT" ]]; then
    echo "quality toolchain drift: expected $QUALITY_EXPECTED_COMMIT, got $QUALITY_ACTUAL_COMMIT" >&2
    exit 1
fi

require_component "$QUALITY_NIGHTLY" miri
require_component "$QUALITY_NIGHTLY" rust-src
require_component "$QUALITY_STABLE" llvm-tools
cargo fuzz --version >/dev/null
cargo llvm-cov --version >/dev/null

CARGO_TARGET_DIR="$QUALITY_TMP/target-stable" \
    cargo metadata --offline --locked --format-version 1 \
    > "$QUALITY_TMP/main-metadata-a.json"
CARGO_TARGET_DIR="$QUALITY_TMP/target-stable" \
    cargo metadata --offline --locked --format-version 1 \
    > "$QUALITY_TMP/main-metadata-b.json"
cmp "$QUALITY_TMP/main-metadata-a.json" "$QUALITY_TMP/main-metadata-b.json"
(
    cd fuzz
    CARGO_TARGET_DIR="$QUALITY_TMP/target-fuzz-metadata" \
        cargo "+$QUALITY_NIGHTLY" metadata --offline --locked --format-version 1 \
        > "$QUALITY_TMP/fuzz-metadata-a.json"
    CARGO_TARGET_DIR="$QUALITY_TMP/target-fuzz-metadata" \
        cargo "+$QUALITY_NIGHTLY" metadata --offline --locked --format-version 1 \
        > "$QUALITY_TMP/fuzz-metadata-b.json"
    cmp "$QUALITY_TMP/fuzz-metadata-a.json" "$QUALITY_TMP/fuzz-metadata-b.json"
)

CARGO_TARGET_DIR="$QUALITY_TMP/target-properties" \
    cargo test --offline --locked --test quality_properties

CARGO_TARGET_DIR="$QUALITY_TMP/target-coverage" \
    cargo llvm-cov --workspace --all-targets --offline --locked \
    --summary-only \
    --fail-under-lines "$QUALITY_MIN_LINE_COVERAGE" \
    --fail-under-functions "$QUALITY_MIN_FUNCTION_COVERAGE" \
    --fail-under-regions "$QUALITY_MIN_REGION_COVERAGE"

CARGO_TARGET_DIR="$QUALITY_TMP/target-miri" \
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check -Zmiri-isolation-error=abort -Zmiri-many-seeds=0..8' \
    cargo "+$QUALITY_NIGHTLY" miri test --lib low_rank_math::tests -- --test-threads=1

CARGO_TARGET_DIR="$QUALITY_TMP/target-asan" \
RUSTFLAGS='-Zsanitizer=address -Cforce-frame-pointers=yes' \
ASAN_OPTIONS='detect_leaks=0:detect_stack_use_after_return=1:halt_on_error=1:strict_string_checks=1' \
    cargo "+$QUALITY_NIGHTLY" test -Zbuild-std --lib \
    --target "$QUALITY_TARGET"

QUALITY_LSAN_RESULT=passed
if probe_lsan_host; then
    CARGO_TARGET_DIR="$QUALITY_TMP/target-lsan" \
    RUSTFLAGS='-Zsanitizer=leak -Cforce-frame-pointers=yes' \
    LSAN_OPTIONS='exitcode=86:halt_on_error=1' \
        cargo "+$QUALITY_NIGHTLY" test -Zbuild-std --lib \
        --target "$QUALITY_TARGET" -- --test-threads=1
else
    lsan_status=$?
    if [[ "$lsan_status" -ne 2 ]]; then
        exit "$lsan_status"
    fi
    QUALITY_LSAN_RESULT=blocked-by-host-ptrace
    echo 'P1 BLOQUEADA PARCIALMENTE: LSan no puede ejecutarse bajo el ptrace de este host.' >&2
    if [[ "$QUALITY_REQUIRE_LSAN" -eq 1 ]]; then
        exit 2
    fi
fi

CARGO_TARGET_DIR="$QUALITY_TMP/target-tsan" \
RUSTFLAGS='-Zsanitizer=thread -Cforce-frame-pointers=yes' \
TSAN_OPTIONS='halt_on_error=1:history_size=7' \
    cargo "+$QUALITY_NIGHTLY" test -Zbuild-std --lib concurrent_ \
    --target "$QUALITY_TARGET" -- --test-threads=1

QUALITY_FUZZ_TMP="$QUALITY_TMP/fuzz-corpus"
QUALITY_FUZZ_PROJECT="$QUALITY_TMP/fuzz-project"
mkdir -p \
    "$QUALITY_FUZZ_TMP/multi-case-solver" \
    "$QUALITY_FUZZ_TMP/persisted-inputs" \
    "$QUALITY_FUZZ_PROJECT" \
    "$QUALITY_TMP/fuzz-artifacts-multi" \
    "$QUALITY_TMP/fuzz-artifacts-persisted"
cp fuzz/seeds/multi-case-solver/finite-values "$QUALITY_FUZZ_TMP/multi-case-solver/"
cp fuzz/seeds/persisted-inputs/empty-object.json "$QUALITY_FUZZ_TMP/persisted-inputs/"
cp fuzz/Cargo.lock fuzz/deny.toml "$QUALITY_FUZZ_PROJECT/"
cp -R fuzz/fuzz_targets "$QUALITY_FUZZ_PROJECT/"
sed "s|path = \"..\"|path = \"$QUALITY_ROOT\"|" \
    fuzz/Cargo.toml > "$QUALITY_FUZZ_PROJECT/Cargo.toml"

(
    cd "$QUALITY_TMP"
    CARGO_TARGET_DIR="$QUALITY_TMP/target-fuzz" \
    ASAN_OPTIONS='detect_leaks=0:halt_on_error=1' \
        cargo "+$QUALITY_NIGHTLY" fuzz run --fuzz-dir "$QUALITY_FUZZ_PROJECT" \
        multi-case-solver "$QUALITY_FUZZ_TMP/multi-case-solver" -- \
        -runs="$QUALITY_FUZZ_RUNS" -rss_limit_mb=768 -max_len=1024 \
        -timeout=10 -artifact_prefix="$QUALITY_TMP/fuzz-artifacts-multi/"
    CARGO_TARGET_DIR="$QUALITY_TMP/target-fuzz" \
    ASAN_OPTIONS='detect_leaks=0:halt_on_error=1' \
        cargo "+$QUALITY_NIGHTLY" fuzz run --fuzz-dir "$QUALITY_FUZZ_PROJECT" \
        persisted-inputs "$QUALITY_FUZZ_TMP/persisted-inputs" -- \
        -runs="$QUALITY_FUZZ_RUNS" -rss_limit_mb=768 -max_len=4096 \
        -timeout=10 -artifact_prefix="$QUALITY_TMP/fuzz-artifacts-persisted/"
)

snapshot_checkout "$QUALITY_TMP/checkout-after"
cmp "$QUALITY_TMP/checkout-before.metadata" "$QUALITY_TMP/checkout-after.metadata"
cmp "$QUALITY_TMP/checkout-before.sha256" "$QUALITY_TMP/checkout-after.sha256"

printf 'P1 SUPERADA: cobertura, Miri, ASan, TSan y fuzzing finalizaron sin modificar el checkout; LSan=%s.\n' \
    "$QUALITY_LSAN_RESULT"
