use cerebro_tidex::representation_evidence::record_representation_evidence;
use cerebro_tidex::security::configured_private_root;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn parse_arguments(args: &[String]) -> Result<&str, &'static str> {
    match args {
        [path] if !path.trim().is_empty() => Ok(path.as_str()),
        _ => Err("usage: record_representation_evidence <sealed-install-request.json>"),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let source_payload = parse_arguments(&args)?;
    let private_root = configured_private_root()?;
    let receipt = record_representation_evidence(&private_root, Path::new(source_payload))?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_arguments_validates_arity() {
        assert!(parse_arguments(&[]).is_err());
        assert!(parse_arguments(&["".into()]).is_err());
        assert!(parse_arguments(&["   ".into()]).is_err());
        assert!(parse_arguments(&["a.json".into(), "b.json".into()]).is_err());
        assert_eq!(
            parse_arguments(&["payload.json".into()]).unwrap(),
            "payload.json"
        );
    }
}
