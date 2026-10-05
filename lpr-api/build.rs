// Builds MNN 3.6.1 (static, CPU) and the C shim in native/, then links both into the binary.
// The first build downloads the MNN source tarball; set MNN_SOURCE_DIR to use a local checkout.
fn main() {
    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/mnn_shim.cpp");
    println!("cargo:rerun-if-env-changed=MNN_SOURCE_DIR");

    let dst = cmake::Config::new("native")
        .profile("Release")
        // MNN's CMakeLists still declares a minimum that CMake 4 refuses without this.
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build_target("lprmnn")
        .build();
    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=lprmnn");
    // MNN registers its operators from static initialisers, so the whole archive must be kept.
    println!("cargo:rustc-link-lib=static:+whole-archive=MNN");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rustc-link-lib={}", if os == "macos" { "c++" } else { "stdc++" });
}
