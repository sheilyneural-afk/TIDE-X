use std::process::Command;

#[test]
fn primary_cli_rejects_unguarded_mutation_commands_before_opening_state() {
    let executable = env!("CARGO_BIN_EXE_cerebro-tidex");
    for (command, expected) in [
        (
            "commit",
            "retired command:commit; use the receipt-bound TIDE-X learning finalizer",
        ),
        (
            "artifact-import-f32",
            "retired command:artifact-import-f32; arbitrary raw delta ingestion is not a governed route",
        ),
        (
            "artifact-ties",
            "retired command:artifact-ties; arbitrary parameter merging is not a governed route",
        ),
    ] {
        let output = Command::new(executable)
            .arg(command)
            .output()
            .expect("retired command should execute the primary binary");
        assert_eq!(output.status.code(), Some(2), "command={command}");
        assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
        assert_eq!(
            String::from_utf8(output.stderr).unwrap().trim(),
            expected,
            "command={command}"
        );
    }
}
