use std::ffi::OsStr;

fn main() {
    let files: Vec<_> = walkdir::WalkDir::new("./Ipopt/src")
        .into_iter()
        .filter_map(|res| {
            res.ok().filter(|entry| {
                entry.path().extension() == Some(OsStr::new("c"))
                    || entry.path().extension() == Some(OsStr::new("cpp"))
                        && !entry.path().ends_with("IpStdJInterface.cpp")
            })
        })
        .map(|entry| entry.path().to_path_buf())
        .collect();
    println!("{:?}", files);
    cc::Build::new()
        .cpp(true)
        .object("/usr/lib/x86_64-linux-gnu/blas/libblas.so")
        .object("/usr/lib/x86_64-linux-gnu/lapack/liblapack.so")
        .includes(&[
            "Ipopt/src/Algorithm",
            "Ipopt/src/Algorithm/Inexact",
            "Ipopt/src/Algorithm/LinearSolvers",
            "Ipopt/src/Apps/AmplSolver",
            "Ipopt/src/Common",
            "Ipopt/src/contrib/CGPenalty",
            "Ipopt/src/Interfaces",
            "Ipopt/src/LinAlg",
            "Ipopt/src/LinAlg/TMatrices",
        ])
        .files(files)
        .compile("Ipopt");
    println!("cargo:rustc-link-search=./Ipopt");
    println!("cargo:rustc-link-lib=static=Ipopt");
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
