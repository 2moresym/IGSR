fn main() {
    cc::Build::new()
        .include("src")
        .files([
            "src/igsr.c",
            "src/jitter.c",
            "src/params.c",
            "src/passes/pass1_reconstruct.c",
            "src/passes/pass2_upsample.c",
            "src/passes/pass3_activate.c",
        ])
        .warnings(true)
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("igsr_core");

    println!("cargo:rerun-if-changed=src/igsr.c");
    println!("cargo:rerun-if-changed=src/jitter.c");
    println!("cargo:rerun-if-changed=src/params.c");
    println!("cargo:rerun-if-changed=src/igsr.h");
    println!("cargo:rerun-if-changed=src/igsr_priv.h");
    println!("cargo:rerun-if-changed=src/passes/pass1_reconstruct.c");
    println!("cargo:rerun-if-changed=src/passes/pass2_upsample.c");
    println!("cargo:rerun-if-changed=src/passes/pass3_activate.c");
}
