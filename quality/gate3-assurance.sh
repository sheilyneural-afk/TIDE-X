#!/usr/bin/env bash
set -euo pipefail
umask 077

QUALITY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QUALITY_TMP=$(mktemp -d /tmp/tidex-gate3-assurance.XXXXXX)
QUALITY_START_EPOCH=$(date +%s)

cleanup() {
    case "$QUALITY_TMP" in
        /tmp/tidex-gate3-assurance.*) find "$QUALITY_TMP" -depth -delete ;;
        *) echo "P3 cleanup refused unexpected path: $QUALITY_TMP" >&2; return 1 ;;
    esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

snapshot_checkout() {
    local prefix=$1
    find . -path ./.git -prune -o -type f -printf '%m\t%s\t%p\0' | LC_ALL=C sort -z > "${prefix}.metadata"
    find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --zero > "${prefix}.sha256"
}

sha_file() { sha256sum "$1" | awk '{print $1}'; }

cd "$QUALITY_ROOT"
QUALITY_HEAD=$(git rev-parse HEAD)
QUALITY_DIFF_SHA256=$(git diff --binary | sha256sum | awk '{print $1}')
QUALITY_STATUS_SHA256=$(git status --porcelain=v1 -z | sha256sum | awk '{print $1}')

if [[ -z "${QUALITY_RECEIPT_PATH:-}" ]]; then
    QUALITY_RECEIPT_PATH="/tmp/tidex-gate3-assurance-${QUALITY_HEAD:0:12}.json"
elif [[ "$QUALITY_RECEIPT_PATH" != /* ]]; then
    QUALITY_RECEIPT_PATH="$QUALITY_ROOT/$QUALITY_RECEIPT_PATH"
fi
case "$QUALITY_RECEIPT_PATH" in
    "$QUALITY_ROOT"|"$QUALITY_ROOT"/*) echo 'P3 receipt must be outside the certified checkout' >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$QUALITY_RECEIPT_PATH")"

snapshot_checkout "$QUALITY_TMP/checkout-before"
QUALITY_CHECKOUT_SHA256=$(sha_file "$QUALITY_TMP/checkout-before.sha256")
QUALITY_METADATA_SHA256=$(sha_file "$QUALITY_TMP/checkout-before.metadata")
git diff --check

bash quality/gate2-verification.sh 2>&1 | tee "$QUALITY_TMP/gate2.log"

mapfile -t REQUIRED_TESTS <<'TESTS'
engine::tests::model_check_engine_authority_lock_prevents_canonical_head_forks
engine::tests::concurrent_engine_authority_serializes_real_canonical_head_advances
engine::tests::model_check_corpus_recovery_is_total_and_fail_closed_for_every_crash_phase
authority::tests::concurrent_directory_moves_never_replace_the_winner
authority::tests::concurrent_idempotent_writers_install_one_complete_object
authority::tests::concurrent_conflicting_writers_never_replace_the_winner
authority::tests::atomic_pointer_replacement_rejects_postrename_inode_substitution
authority::tests::replacement_after_open_fails_closed_without_accepting_replacement_bytes
authority::tests::mutation_race_on_the_open_inode_fails_closed
authority::tests::transactional_move_never_overwrites_and_recovers_its_own_link
engine::tests::recover_rolls_back_an_intent_that_never_retired_the_prior_corpus
engine::tests::recover_restores_archived_prior_corpus
engine::tests::recover_requires_replay_when_new_corpus_already_published
knowledge_engine::tests::canonical_head_rejects_double_advance_concurrently
ledger::tests::duplicate_v2_payload_key_is_ambiguous_not_first_match
engine::tests::support_controller_execution_receipt_and_ledger_invariants
engine::tests::transition_commit_after_verified_corpus_transition_invariants
TESTS

CARGO_TARGET_DIR="$QUALITY_TMP/target-p3" cargo test --lib --offline --locked -- --list > "$QUALITY_TMP/test-list.txt"
for test_name in "${REQUIRED_TESTS[@]}"; do
    grep -Fxq "${test_name}: test" "$QUALITY_TMP/test-list.txt" || { echo "P3 required assurance test missing: $test_name" >&2; exit 1; }
done

: > "$QUALITY_TMP/p3-tests.log"
: > "$QUALITY_TMP/p3-tests-passed.txt"
for test_name in "${REQUIRED_TESTS[@]}"; do
    printf 'P3 RUN %s\n' "$test_name" | tee -a "$QUALITY_TMP/p3-tests.log"
    CARGO_TARGET_DIR="$QUALITY_TMP/target-p3" cargo test --lib --offline --locked "$test_name" -- --exact --test-threads=1 2>&1 | tee -a "$QUALITY_TMP/p3-tests.log"
    printf '%s\n' "$test_name" >> "$QUALITY_TMP/p3-tests-passed.txt"
done

CARGO_TARGET_DIR="$QUALITY_TMP/target-p3" cargo test --lib --offline --locked engine::tests::model_check_ -- --test-threads=1 2>&1 | tee -a "$QUALITY_TMP/p3-tests.log"

snapshot_checkout "$QUALITY_TMP/checkout-after"
cmp "$QUALITY_TMP/checkout-before.metadata" "$QUALITY_TMP/checkout-after.metadata"
cmp "$QUALITY_TMP/checkout-before.sha256" "$QUALITY_TMP/checkout-after.sha256"
git diff --check

QUALITY_END_EPOCH=$(date +%s)
QUALITY_GATE2_SHA256=$(sha_file quality/gate2-verification.sh)
QUALITY_GATE3_SHA256=$(sha_file quality/gate3-assurance.sh)
QUALITY_GATE2_LOG_SHA256=$(sha_file "$QUALITY_TMP/gate2.log")
QUALITY_P3_LOG_SHA256=$(sha_file "$QUALITY_TMP/p3-tests.log")
QUALITY_STABLE_RUSTC=$(rustc -Vv)
QUALITY_NIGHTLY_RUSTC=$(rustc +nightly -Vv)

python3 - \
    "$QUALITY_RECEIPT_PATH" \
    "$QUALITY_TMP/gate2.log" \
    "$QUALITY_TMP/p3-tests-passed.txt" \
    "$QUALITY_HEAD" \
    "$QUALITY_DIFF_SHA256" \
    "$QUALITY_STATUS_SHA256" \
    "$QUALITY_CHECKOUT_SHA256" \
    "$QUALITY_METADATA_SHA256" \
    "$QUALITY_GATE2_SHA256" \
    "$QUALITY_GATE3_SHA256" \
    "$QUALITY_GATE2_LOG_SHA256" \
    "$QUALITY_P3_LOG_SHA256" \
    "$QUALITY_START_EPOCH" \
    "$QUALITY_END_EPOCH" \
    "$QUALITY_STABLE_RUSTC" \
    "$QUALITY_NIGHTLY_RUSTC" <<'PY'
import json
import pathlib
import re
import sys

(
    receipt_path,
    gate2_log_path,
    passed_path,
    head,
    diff_sha,
    status_sha,
    checkout_sha,
    metadata_sha,
    gate2_sha,
    gate3_sha,
    gate2_log_sha,
    p3_log_sha,
    start_epoch,
    end_epoch,
    stable_rustc,
    nightly_rustc,
) = sys.argv[1:]

text = pathlib.Path(gate2_log_path).read_text(errors="replace")
global_match = re.search(
    r"P2 global coverage: lines=([0-9.]+)% functions=([0-9.]+)% regions=([0-9.]+)%",
    text,
)
if not global_match:
    raise SystemExit("P3 could not recover P2 global coverage from Gate2 evidence")
critical = {}
for suffix, line, function, region in re.findall(
    r"P2 coverage ([^:]+): lines=([0-9.]+)% functions=([0-9.]+)% regions=([0-9.]+)%",
    text,
):
    critical[suffix] = {
        "lines": float(line),
        "functions": float(function),
        "regions": float(region),
    }
if len(critical) < 6:
    raise SystemExit("P3 Gate2 log lacks the six critical coverage records")

passed = [line for line in pathlib.Path(passed_path).read_text().splitlines() if line]
receipt = {
    "schema": "cerebro.tidex.quality_assurance_receipt/v1",
    "gate": "P3",
    "result": "passed",
    "head_commit": head,
    "working_diff_sha256": diff_sha,
    "working_status_sha256": status_sha,
    "checkout_manifest_sha256": checkout_sha,
    "checkout_metadata_sha256": metadata_sha,
    "started_epoch": int(start_epoch),
    "finished_epoch": int(end_epoch),
    "duration_seconds": int(end_epoch) - int(start_epoch),
    "toolchains": {
        "stable_rustc_vv": stable_rustc,
        "nightly_rustc_vv": nightly_rustc,
    },
    "evidence_sha256": {
        "gate2_script": gate2_sha,
        "gate3_script": gate3_sha,
        "gate2_log": gate2_log_sha,
        "p3_tests_log": p3_log_sha,
    },
    "p2_coverage": {
        "global": {
            "lines": float(global_match.group(1)),
            "functions": float(global_match.group(2)),
            "regions": float(global_match.group(3)),
        },
        "critical": critical,
    },
    "p3_model_scope": {
        "canonical_head_writers": 2,
        "canonical_head_lock_modes": ["locked", "deliberately_unlocked_negative_control"],
        "unsealed_recovery_state_combinations": 8,
        "explicit_crash_phases_without_receipt": 5,
        "receipt_sealed_path": "authenticated_receipt_recovery",
    },
    "required_assurance_tests": passed,
    "limitations": [
        "bounded deterministic state-machine model checking",
        "not a universal proof of kernel/filesystem/hardware behavior",
        "loom not added without an offline reproducible pinned dependency",
    ],
}
path = pathlib.Path(receipt_path)
path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
PY

sha256sum "$QUALITY_RECEIPT_PATH" > "${QUALITY_RECEIPT_PATH}.sha256"
printf 'P3 SUPERADA: Gate2 acumulativa, model checking acotado, recuperación adversarial y concurrencia pasan sin modificar el checkout. receipt=%s\n' "$QUALITY_RECEIPT_PATH"
