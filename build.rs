fn main() -> miette::Result<()> {
    let mut b = autocxx_build::Builder::new(
        "src/lib.rs",
        &[
            "Ipopt/src/Algorithm",
            "Ipopt/src/Apps",
            "Ipopt/src/Common",
            "Ipopt/src/contrib",
            "Ipopt/src/Interfaces",
            "Ipopt/src/LinAlg",
        ],
    )
    .build()?;
    // This assumes all your C++ bindings are in main.rs
    b.flag_if_supported("-std=c++14").compile("minxent"); // arbitrary library name, pick anything
    println!("cargo:rerun-if-changed=src/lib.rs");
    // Add instructions to link to any C++ libraries you need.
    Ok(())
}
