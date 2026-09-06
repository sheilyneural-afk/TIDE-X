#!/usr/bin/env bash
set -euo pipefail

umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QUALITY_NIGHTLY=${QUALITY_NIGHTLY:-nightly}
QUALITY_EXPECTED_COMMIT=${QUALITY_EXPECTED_COMMIT:-0ed41eb4142dda2df61eb1145a312c1a9d62eb56}
QUALITY_TARGET=${QUALITY_TARGET:-x86_64-unknown-linux-gnu}
QUALITY_FUZZ_RUNS=${QUALITY_FUZZ_RUNS:-100000}
QUALITY_REQUIRE_LSAN=1

QUALITY_LINE_FLOOR=82
QUALITY_FUNCTION_FLOOR=75
QUALITY_REGION_FLOOR=80
QUALITY_MIN_LINE_COVERAGE=${QUALITY_MIN_LINE_COVERAGE:-$QUALITY_LINE_FLOOR}
QUALITY_MIN_FUNCTION_COVERAGE=${QUALITY_MIN_FUNCTION_COVERAGE:-$QUALITY_FUNCTION_FLOOR}
QUALITY_MIN_REGION_COVERAGE=${QUALITY_MIN_REGION_COVERAGE:-$QUALITY_REGION_FLOOR}

QUALITY_TMP=$(mktemp -d /tmp/tidex-gate2-verification.XXXXXX)

cleanup() {
    case "$QUALITY_TMP" in
        /tmp/tidex-gate2-verification.*) find "$QUALITY_TMP" -depth -delete ;;
        *)
            echo "P2 cleanup refused unexpected path: $QUALITY_TMP" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

require_integer() {
    local name=$1 value=$2
    case "$value" in
        ''|*[!0-9]*) echo "$name must be an integer" >&2; exit 2 ;;
    esac
}

require_integer QUALITY_FUZZ_RUNS "$QUALITY_FUZZ_RUNS"
require_integer QUALITY_MIN_LINE_COVERAGE "$QUALITY_MIN_LINE_COVERAGE"
require_integer QUALITY_MIN_FUNCTION_COVERAGE "$QUALITY_MIN_FUNCTION_COVERAGE"
require_integer QUALITY_MIN_REGION_COVERAGE "$QUALITY_MIN_REGION_COVERAGE"
if (( QUALITY_FUZZ_RUNS < 100000 || QUALITY_FUZZ_RUNS > 10000000 )); then
    echo 'QUALITY_FUZZ_RUNS must be between 100000 and 10000000 for P2' >&2
    exit 2
fi
if (( QUALITY_MIN_LINE_COVERAGE < QUALITY_LINE_FLOOR || QUALITY_MIN_LINE_COVERAGE > 100 )); then
    echo "QUALITY_MIN_LINE_COVERAGE must be between $QUALITY_LINE_FLOOR and 100 for P2" >&2
    exit 2
fi
if (( QUALITY_MIN_FUNCTION_COVERAGE < QUALITY_FUNCTION_FLOOR || QUALITY_MIN_FUNCTION_COVERAGE > 100 )); then
    echo "QUALITY_MIN_FUNCTION_COVERAGE must be between $QUALITY_FUNCTION_FLOOR and 100 for P2" >&2
    exit 2
fi
if (( QUALITY_MIN_REGION_COVERAGE < QUALITY_REGION_FLOOR || QUALITY_MIN_REGION_COVERAGE > 100 )); then
    echo "QUALITY_MIN_REGION_COVERAGE must be between $QUALITY_REGION_FLOOR and 100 for P2" >&2
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

cd "$QUALITY_ROOT"
snapshot_checkout "$QUALITY_TMP/checkout-before"

git diff --check
QUALITY_ACTUAL_COMMIT=$(rustc "+$QUALITY_NIGHTLY" -Vv | sed -n 's/^commit-hash: //p')
if [[ "$QUALITY_ACTUAL_COMMIT" != "$QUALITY_EXPECTED_COMMIT" ]]; then
    echo "P2 toolchain drift: expected $QUALITY_EXPECTED_COMMIT, got $QUALITY_ACTUAL_COMMIT" >&2
    exit 1
fi
require_component "$QUALITY_NIGHTLY" miri
require_component "$QUALITY_NIGHTLY" rust-src
cargo llvm-cov --version >/dev/null

# P2 is cumulative: lower release/tooling gates cannot be bypassed.
bash quality/gate0-release.sh
QUALITY_FUZZ_RUNS="$QUALITY_FUZZ_RUNS" \
QUALITY_REQUIRE_LSAN="$QUALITY_REQUIRE_LSAN" \
QUALITY_MIN_LINE_COVERAGE="$QUALITY_MIN_LINE_COVERAGE" \
QUALITY_MIN_FUNCTION_COVERAGE="$QUALITY_MIN_FUNCTION_COVERAGE" \
QUALITY_MIN_REGION_COVERAGE="$QUALITY_MIN_REGION_COVERAGE" \
    bash quality/gate1-tooling.sh

# Recompute one authoritative JSON report for both global and per-file floors.
CARGO_TARGET_DIR="$QUALITY_TMP/target-coverage" \
    cargo llvm-cov --workspace --all-targets --offline --locked \
    --json --output-path "$QUALITY_TMP/coverage.json"

python3 - \
    "$QUALITY_TMP/coverage.json" \
    "$QUALITY_MIN_LINE_COVERAGE" \
    "$QUALITY_MIN_FUNCTION_COVERAGE" \
    "$QUALITY_MIN_REGION_COVERAGE" <<'PY'
import json
import pathlib
import sys

coverage_path = pathlib.Path(sys.argv[1])
min_lines = float(sys.argv[2])
min_functions = float(sys.argv[3])
min_regions = float(sys.argv[4])

data = json.loads(coverage_path.read_text())["data"][0]
totals = data["totals"]
observed = {
    "lines": float(totals["lines"]["percent"]),
    "functions": float(totals["functions"]["percent"]),
    "regions": float(totals["regions"]["percent"]),
}
required_global = {
    "lines": min_lines,
    "functions": min_functions,
    "regions": min_regions,
}
for metric, required in required_global.items():
    if observed[metric] + 1e-12 < required:
        raise SystemExit(
            f"P2 global {metric} coverage {observed[metric]:.2f}% < {required:.2f}%"
        )

critical_line_floors = {
    "src/engine/runtime.rs": 80.0,
    "src/engine/transition.rs": 75.0,
    "src/engine/support.rs": 80.0,
    "src/engine/analysis.rs": 85.0,
    "src/isolated_execution.rs": 85.0,
    "src/digest.rs": 95.0,
}
files = data["files"]
for suffix, required in critical_line_floors.items():
    matches = [entry for entry in files if entry["filename"].endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"P2 coverage file resolution failed for {suffix}: {len(matches)} matches")
    summary = matches[0]["summary"]
    line_percent = float(summary["lines"]["percent"])
    function_percent = float(summary["functions"]["percent"])
    region_percent = float(summary["regions"]["percent"])
    print(
        f"P2 coverage {suffix}: lines={line_percent:.2f}% "
        f"functions={function_percent:.2f}% regions={region_percent:.2f}%"
    )
    if line_percent + 1e-12 < required:
        raise SystemExit(
            f"P2 critical line coverage {suffix} {line_percent:.2f}% < {required:.2f}%"
        )

print(
    "P2 global coverage: "
    f"lines={observed['lines']:.2f}% functions={observed['functions']:.2f}% "
    f"regions={observed['regions']:.2f}%"
)
PY

# P2 repeats the zero-warning compiler gate after the strict coverage build.
CARGO_TARGET_DIR="$QUALITY_TMP/target-clippy" \
    cargo clippy --all-targets --offline --locked -- -D warnings

MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check -Zmiri-isolation-error=abort -Zmiri-many-seeds=0..8'
export MIRIFLAGS
for module in low_rank_math linalg trust_region transport; do
    CARGO_TARGET_DIR="$QUALITY_TMP/target-miri" \
        cargo "+$QUALITY_NIGHTLY" miri test --lib "${module}::tests" -- --test-threads=1
    printf 'P2 Miri OK: %s\n' "$module"
done

# P1 intentionally filters TSan to concurrency-named tests. P2 removes that
# naming dependency and instruments every test target in the workspace.
CARGO_TARGET_DIR="$QUALITY_TMP/target-tsan-full" \
RUSTFLAGS='-Zsanitizer=thread -Cforce-frame-pointers=yes' \
TSAN_OPTIONS='halt_on_error=1:second_deadlock_stack=1' \
    cargo "+$QUALITY_NIGHTLY" test -Zbuild-std --all-targets \
    --offline --locked --target "$QUALITY_TARGET" -- --test-threads=1

snapshot_checkout "$QUALITY_TMP/checkout-after"
cmp "$QUALITY_TMP/checkout-before.metadata" "$QUALITY_TMP/checkout-after.metadata"
cmp "$QUALITY_TMP/checkout-before.sha256" "$QUALITY_TMP/checkout-after.sha256"

git diff --check
printf 'P2 SUPERADA: P0+P1, cobertura crítica, Clippy, Miri ampliado y TSan completo pasan sin modificar el checkout.\n'
