use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

const DIGEST_DOMAIN: &[u8] = b"CEREBRO:TIDEX:COMPILED-INPUTS:v3\0";

#[derive(Clone, Copy)]
enum CanonicalInput {
    Directory {
        path: &'static str,
        role: &'static str,
    },
    File {
        path: &'static str,
        role: &'static str,
    },
}

// This is deliberately a closed, typed list. TIDEX_SOURCE_TREE_DIGEST is the
// identity of the compiled crate/release configuration, not a digest of the
// checkout. Donors, corpora, runtime state, evaluation data, documentation,
// noncompiled quality scripts and fuzz targets have independent authorities and
// must never change this value. Explicit Cargo binary source roots are included
// even when they live outside src; their location does not make them data.
const CANONICAL_COMPILED_INPUTS: &[CanonicalInput] = &[
    CanonicalInput::Directory {
        path: "src",
        role: "crate-source-tree",
    },
    CanonicalInput::File {
        path: "Cargo.toml",
        role: "crate-manifest",
    },
    CanonicalInput::File {
        path: "Cargo.lock",
        role: "resolved-dependency-graph",
    },
    CanonicalInput::File {
        path: "build.rs",
        role: "build-script",
    },
    CanonicalInput::File {
        path: "rust-toolchain.toml",
        role: "pinned-rust-toolchain",
    },
    CanonicalInput::File {
        path: "deny.toml",
        role: "release-dependency-policy",
    },
    CanonicalInput::File {
        path: ".cargo/config.toml",
        role: "cargo-build-configuration",
    },
];

fn require_directory(path: &Path, role: &str) {
    let metadata = fs::symlink_metadata(path).unwrap_or_else(|error| {
        panic!(
            "required canonical {role} directory is missing or unreadable: {}: {error}",
            path.display()
        )
    });
    assert!(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        "required canonical {role} directory must be a real directory, not a symlink: {}",
        path.display()
    );
}

fn require_regular_file(path: &Path, role: &str) {
    let metadata = fs::symlink_metadata(path).unwrap_or_else(|error| {
        panic!(
            "required canonical {role} file is missing or unreadable: {}: {error}",
            path.display()
        )
    });
    assert!(
        !metadata.file_type().is_symlink() && metadata.is_file(),
        "required canonical {role} input must be a regular file, not a symlink: {}",
        path.display()
    );
}

fn reject_competing_input(path: &Path, description: &str) {
    if fs::symlink_metadata(path).is_ok() {
        panic!(
            "ambiguous canonical build input: {description} exists at {}; use only the declared canonical input",
            path.display()
        );
    }
}

fn canonical_relative_path(path: &Path) -> String {
    let components = path
        .components()
        .map(|component| match component {
            Component::Normal(segment) => {
                let segment = segment.to_str().unwrap_or_else(|| {
                    panic!(
                        "canonical input path is not valid UTF-8: {}",
                        path.display()
                    )
                });
                assert!(
                    !segment.contains('\\'),
                    "canonical input path contains a platform-ambiguous separator: {}",
                    path.display()
                );
                segment
            }
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                panic!("canonical input path is not normalized: {}", path.display())
            }
        })
        .collect::<Vec<_>>();
    assert!(
        !components.is_empty(),
        "canonical input path must not be empty"
    );
    components.join("/")
}

fn collect_regular_files(directory: &Path, role: &str, out: &mut Vec<(String, String)>) {
    require_directory(directory, role);
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| {
            panic!(
                "required canonical {role} directory cannot be enumerated: {}: {error}",
                directory.display()
            )
        })
        .map(|entry| {
            entry.unwrap_or_else(|error| {
                panic!(
                    "required canonical {role} directory entry is unreadable: {}: {error}",
                    directory.display()
                )
            })
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| canonical_relative_path(&entry.path()));

    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).unwrap_or_else(|error| {
            panic!(
                "canonical {role} entry metadata is unreadable: {}: {error}",
                path.display()
            )
        });
        assert!(
            !metadata.file_type().is_symlink(),
            "canonical {role} tree must not contain symlinks: {}",
            path.display()
        );
        if metadata.is_dir() {
            collect_regular_files(&path, role, out);
        } else if metadata.is_file() {
            out.push((canonical_relative_path(&path), role.to_owned()));
        } else {
            panic!(
                "canonical {role} tree must contain only regular files and directories: {}",
                path.display()
            );
        }
    }
}

fn manifest_literal(value: &str, label: &str) -> String {
    let value = value.trim();
    let quote = value
        .chars()
        .next()
        .unwrap_or_else(|| panic!("empty {label}"));
    assert!(
        quote == '"' || quote == '\'',
        "canonical {label} must be a quoted single-line literal"
    );
    let rest = &value[quote.len_utf8()..];
    let end = rest
        .find(quote)
        .unwrap_or_else(|| panic!("unterminated {label}"));
    let literal = &rest[..end];
    let suffix = rest[end + quote.len_utf8()..].trim();
    assert!(
        (suffix.is_empty() || suffix.starts_with('#'))
            && !literal.is_empty()
            && !literal.contains('\\')
            && !literal.chars().any(char::is_control),
        "canonical {label} must not use escapes, multiline strings or trailing values"
    );
    literal.to_owned()
}

fn finish_binary_path(path: &mut Option<String>, binaries: &mut Vec<String>) {
    let path = path
        .take()
        .unwrap_or_else(|| panic!("every canonical Cargo [[bin]] must declare an explicit path"));
    let canonical = canonical_relative_path(Path::new(&path));
    assert!(
        Path::new(&canonical)
            .extension()
            .is_some_and(|extension| extension == "rs"),
        "canonical binary entry point must be a Rust source file: {canonical}"
    );
    binaries.push(canonical);
}

// The crate deliberately uses explicit target declarations (autobins=false).
// Read their canonical literal paths without spawning Cargo recursively or
// introducing another build dependency. Unsupported manifest syntax fails
// closed instead of silently omitting a compiled binary from the identity.
fn declared_binary_paths(manifest: &str) -> Vec<String> {
    let mut in_package = false;
    let mut in_binary = false;
    let mut automatic_binaries_disabled = false;
    let mut binary_path = None;
    let mut binaries = Vec::new();
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            if in_binary {
                finish_binary_path(&mut binary_path, &mut binaries);
            }
            let header = line.split('#').next().unwrap_or("").trim();
            if header.starts_with("[[") {
                let table = header
                    .strip_prefix("[[")
                    .and_then(|s| s.strip_suffix("]]"))
                    .unwrap_or_else(|| panic!("noncanonical Cargo target table: {header}"))
                    .trim();
                // Reject unknown/escaped array-table spellings: a permissive
                // scanner must not mistake a binary declaration for data.
                assert!(
                    matches!(table, "bin" | "test" | "bench" | "example"),
                    "unsupported canonical Cargo array table: {header}"
                );
                in_binary = table == "bin";
            } else {
                in_binary = false;
            }
            in_package = header == "[package]";
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if in_package && key == "autobins" {
            assert!(
                !automatic_binaries_disabled,
                "duplicate canonical package.autobins declaration"
            );
            assert!(
                value.split('#').next().unwrap_or("").trim() == "false",
                "canonical binary source discovery requires package.autobins=false"
            );
            automatic_binaries_disabled = true;
        }
        if in_binary && key == "path" {
            assert!(binary_path.is_none(), "duplicate canonical binary path");
            binary_path = Some(manifest_literal(value, "Cargo binary path"));
        }
    }
    if in_binary {
        finish_binary_path(&mut binary_path, &mut binaries);
    }
    assert!(
        automatic_binaries_disabled,
        "canonical binary source discovery requires explicit package.autobins=false"
    );
    binaries
}

fn collect_declared_binary_sources(files: &mut Vec<(String, String)>) {
    const ROLE: &str = "crate-binary-source-tree";
    let manifest = fs::read_to_string("Cargo.toml")
        .unwrap_or_else(|error| panic!("cannot read canonical Cargo manifest: {error}"));
    let binaries = declared_binary_paths(&manifest);
    let mut paths = files
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<BTreeSet<_>>();
    let mut roots = BTreeSet::new();
    for binary in binaries {
        let path = Path::new(&binary);
        // Validate every parent component so a newly declared source root
        // cannot traverse a symlink in an intermediate directory.
        let mut prefix = std::path::PathBuf::new();
        for component in path.parent().into_iter().flat_map(Path::components) {
            prefix.push(component.as_os_str());
            require_directory(&prefix, ROLE);
        }
        require_regular_file(path, ROLE);
        if path.starts_with("src") {
            continue;
        }
        if paths.insert(binary.clone()) {
            files.push((binary.clone(), ROLE.to_owned()));
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            roots.insert(canonical_relative_path(parent));
        }
    }
    for root in roots {
        let mut local_sources = Vec::new();
        collect_regular_files(Path::new(&root), ROLE, &mut local_sources);
        println!("cargo:rerun-if-changed={root}");
        // Rust modules adjacent to an external bin are also compile inputs.
        // Python experiments, receipts and documentation retain their separate
        // authorities even when they share the same source directory.
        for (path, role) in local_sources {
            if Path::new(&path)
                .extension()
                .is_some_and(|extension| extension == "rs")
                && paths.insert(path.clone())
            {
                files.push((path, role));
            }
        }
    }
}

fn update_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn hash_regular_file(hasher: &mut Sha256, path: &str, role: &str) {
    let filesystem_path = Path::new(path);
    require_regular_file(filesystem_path, role);
    let bytes = fs::read(filesystem_path).unwrap_or_else(|error| {
        panic!(
            "canonical {role} input cannot be read: {}: {error}",
            filesystem_path.display()
        )
    });

    // Every record is length framed and has an explicit type, role, relative
    // path and content. No concatenation ambiguity can make two distinct
    // canonical input sets hash as the same structured stream.
    hasher.update([1_u8]); // regular-file record
    update_field(hasher, b"regular-file");
    update_field(hasher, role.as_bytes());
    update_field(hasher, path.as_bytes());
    update_field(hasher, &bytes);
}

fn main() {
    // Cargo recognizes these legacy alternatives. Rejecting them prevents an
    // undeclared file from influencing the compiler/toolchain while escaping
    // the declared canonical identity.
    reject_competing_input(Path::new("rust-toolchain"), "legacy rust-toolchain file");
    reject_competing_input(
        Path::new(".cargo/config"),
        "legacy Cargo configuration file",
    );
    require_directory(Path::new(".cargo"), "Cargo configuration root");

    let mut files = Vec::new();
    for input in CANONICAL_COMPILED_INPUTS {
        match input {
            CanonicalInput::Directory { path, role } => {
                let root = Path::new(path);
                collect_regular_files(root, role, &mut files);

                // This detects additions or removals that cannot appear in a
                // prior per-file watch, while remaining scoped to `src`.
                println!("cargo:rerun-if-changed={path}");
            }
            CanonicalInput::File { path, role } => {
                require_regular_file(Path::new(path), role);
                files.push((canonical_relative_path(Path::new(path)), (*role).to_owned()));
            }
        }
    }
    collect_declared_binary_sources(&mut files);
    files.sort_unstable();
    assert!(
        files.windows(2).all(|pair| pair[0].0 != pair[1].0),
        "canonical compiled inputs contain a duplicate path"
    );

    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    for (path, role) in files {
        println!("cargo:rerun-if-changed={path}");
        hash_regular_file(&mut hasher, &path, &role);
    }
    let digest = format!("{:x}", hasher.finalize());
    println!("cargo:rustc-env=TIDEX_SOURCE_TREE_DIGEST={digest}");
}
