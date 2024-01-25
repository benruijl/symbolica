use std::process::Command;
fn main() {
    let output = Command::new("git")
        .args(["describe", "--tags"])
        .output()
        .expect("Could not run git command. Is it installed?");
    let git_desc = String::from_utf8(output.stdout).unwrap();
    println!("cargo:rustc-env=SYMBOLICA_VERSION={}", git_desc);
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs")
}
