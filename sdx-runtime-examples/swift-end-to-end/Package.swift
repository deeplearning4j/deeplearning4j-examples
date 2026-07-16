// swift-tools-version: 5.9
import PackageDescription

// End-to-end SDX runtime example. The path dependency assumes a sibling
// `deeplearning4j` checkout next to this examples repository; when building
// against an unpacked SDK package, point it at <sdk>/wrappers/swift instead.
//
// Two targets are provided:
//
//  SdxEndToEnd   — DSP runtime (dsp_runtime_c.h) example; links nd4jcpu.
//  SdxLlmExample — LLM AOT runtime (sdx_llm_c.h) example; links sdx_llm.
//
// Build the LLM example:
//   export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
//   swift run -Xcc -I$SDX_LLM_AOT_HOME/include \
//             -Xlinker -L$SDX_LLM_AOT_HOME/lib \
//             SdxLlmExample \
//             ~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//             ~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json
let package = Package(
    name: "SdxEndToEnd",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(path: "../../../deeplearning4j/libnd4j/include/dsp/runtime/bindings/swift")
    ],
    targets: [
        // ── DSP runtime example (existing) ────────────────────────────────────
        .executableTarget(
            name: "SdxEndToEnd",
            dependencies: [
                .product(name: "SdxRuntime", package: "swift")
            ],
            path: "Sources/SdxEndToEnd"
        ),

        // ── LLM AOT runtime: C system library shim ────────────────────────────
        //
        // CSdxLlm is a system-library target that exposes sdx_llm_c.h to Swift
        // via a module map.  The header is resolved by Sources/CSdxLlm/shim.h:
        //   • Pass -Xcc -I$SDX_LLM_AOT_HOME/include   (unpacked SDK), OR
        //   • Use the source-tree path (sibling deeplearning4j checkout).
        .systemLibrary(
            name: "CSdxLlm",
            path: "Sources/CSdxLlm"
        ),

        // ── LLM AOT runtime: Swift wrapper library ─────────────────────────────
        .target(
            name: "SdxLlm",
            dependencies: ["CSdxLlm"],
            path: "Sources/SdxLlm"
        ),

        // ── LLM AOT runtime: runnable example ────────────────────────────────
        .executableTarget(
            name: "SdxLlmExample",
            dependencies: ["SdxLlm"],
            path: "Sources/SdxLlmExample"
        ),
    ]
)
