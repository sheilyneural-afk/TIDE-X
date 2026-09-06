fn main() {
    if let Err(error) = cerebro_tidex::pure_capability_e2e::run_pure_linear_runner() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
