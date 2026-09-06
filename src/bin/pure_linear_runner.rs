fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), cerebro_tidex::BrainError> {
    cerebro_tidex::pure_capability_e2e::run_pure_linear_runner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_linear_runner_fails_closed_without_container_input() {
        assert!(run().is_err());
    }
}
