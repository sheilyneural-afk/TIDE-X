use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn main() {
    let mut files = Vec::new();
    collect_rs(Path::new("src"), &mut files);
    for extra in ["build.rs", "Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        let path = PathBuf::from(extra);
        if path.exists() {
            files.push(path);
        }
    }
    files.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:SOURCE-TREE:v1\0");
    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        let rel = path.to_string_lossy();
        let bytes = fs::read(path).expect("source tree file must be readable at build time");
        hasher.update((rel.len() as u64).to_be_bytes());
        hasher.update(rel.as_bytes());
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(&bytes);
    }
    let digest = format!("{:x}", hasher.finalize());
    println!("cargo:rustc-env=TIDEX_SOURCE_TREE_DIGEST={digest}");
}
