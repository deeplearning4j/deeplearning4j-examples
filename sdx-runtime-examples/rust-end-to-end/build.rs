// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

//! Build script — adds the link search path for both the SDX runtime library
//! (`end_to_end` binary) and the SDX LLM library (`llm` binary).
//!
//! Search order for the SDX runtime library:
//!   1. `SDX_RUNTIME_LIB_DIR` env var (as per the `sdx-runtime` crate's own build.rs)
//!   2. `../../lib` relative to this crate (unpacked SDK layout)
//!
//! Search order for the SDX LLM library:
//!   1. `SDX_LLM_LIB_DIR` env var
//!   2. `SDX_LLM_AOT_HOME/lib` env var
//!   3. `../../lib` relative to this crate (same SDK layout convention)

use std::env;
use std::path::PathBuf;

fn main() {
    // ── SDX LLM library (libsdx_llm.so) ─────────────────────────────────────
    //
    // The `llm` binary links `sdx_llm` via `#[link(name = "sdx_llm")]`
    // inside `src/sdx_llm.rs`.  Cargo needs the directory on its native
    // search path.

    let llm_lib_dir = env::var("SDX_LLM_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|_| {
            env::var("SDX_LLM_AOT_HOME").map(|h| PathBuf::from(h).join("lib"))
        })
        .unwrap_or_else(|_| {
            let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
            PathBuf::from(manifest).join("../../lib")
        });

    if llm_lib_dir.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            llm_lib_dir.canonicalize()
                .unwrap_or(llm_lib_dir.clone())
                .display()
        );
    }

    // Rebuild if these env vars change.
    println!("cargo:rerun-if-env-changed=SDX_LLM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SDX_LLM_AOT_HOME");
    println!("cargo:rerun-if-env-changed=SDX_RUNTIME_LIB_DIR");
}
