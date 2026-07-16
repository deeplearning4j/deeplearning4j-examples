# SDX Runtime Examples

Basic usage examples for the SDX Runtime language bindings.
Each subdirectory contains a minimal runnable program that creates
an `SdxRuntime`, loads a model, and runs inference.

| Language | File                        |
|----------|-----------------------------|
| Java     | `java/BasicUsage.java`      |
| Kotlin   | `kotlin/BasicUsage.kt`      |
| Python   | `python/basic_usage.py`     |
| Rust     | `rust/basic_usage.rs`       |
| C#       | `csharp/BasicUsage.cs`      |
| Swift    | `swift/BasicUsage.swift`    |

These examples target SDX Runtime ABI version 1. Backend selectors mirror the
native execution modes, including HIP graph replay (`9`) and Vulkan command-
buffer replay (`11`), with strict selection available through each wrapper's
model/session options.

## End-to-end examples (one directory per language)

Each `*-end-to-end/` directory is a self-contained project that walks the
whole SDK lifecycle against a real model:

- input-contract discovery (`sdxGetNumInputs` / `sdxGetInputName` —
  external inputs cover the model's constants, variables/weights, AND
  placeholders, bound positionally),
- placeholder marking, warmup runs, `sdxFreezeShapes`, and the DSP
  replay fast path,
- execution-report telemetry (plan phase, execution count, applied
  backend, fallback flag, wall time),
- caller-provided output buffers verified against the model's canonical
  expected values, and the `sdxGetLastError` error path.

Every example uses an idiomatic wrapper API following its ecosystem's
established ML-runtime conventions (onnxruntime's per-language APIs, CoreML,
react-native-executorch, etc.) — named-input inference with the positional
C-ABI contract hidden behind input-name discovery, the platform's natural
tensor type, structured telemetry DTOs, and the language's canonical resource
management:

| Language | Directory | Build/run | Idiomatic surface |
|----------|-----------|-----------|-------------------|
| Java | `java-end-to-end/` | `mvn -q compile exec:java` | ORT-style `SdxEnvironment`/`SdxSession`/`SdxTensor` client package, try-with-resources, named-input `Map` |
| Python | `python-end-to-end/` | `python3 end_to_end.py` | numpy-first `run_named(feed_dict)`, `get_inputs()` metadata, frozen `ExecutionSummary` dataclass, context managers |
| Rust | `rust-end-to-end/` | `cargo run --release` | `ndarray` feature (`run_named_shaped` with `ArrayViewD<f32>`), builder flow, `#[non_exhaustive]` error enum |
| C# | `csharp-end-to-end/` | `dotnet run` | `DenseTensor<float>` named-input `Run`, `record ExecutionReport`, using-declarations, `#nullable enable` |
| Kotlin | `kotlin-end-to-end/` | `gradle run` | `runNamed(map)`, `FloatTensor`, typed `PlanPhase`/`SdxBackend` enums, data classes, `use {}` |
| Swift | `swift-end-to-end/` | `swift run SdxEndToEnd` | value-type `SdxTensor`, `[String: SdxTensor]` run, typed enums + `CustomStringConvertible` report |
| TypeScript / Node.js | `typescript/node/` | `npm install && npm start` | ORT-node-style classes, `Tensor {data: Float32Array, dims}`, `Symbol.dispose`/`using` |
| TypeScript / React Native | `typescript/react-native/` | see its README | TurboModule spec + `SdxSession` class + `useSdxModel` hook over the Android JNI / iOS ObjC++ bridge |
| WebAssembly | `typescript/wasm/` | `npm install && npm run start:mock` | onnxruntime-web-style wrapper over an Emscripten `MODULARIZE` module (MEMFS model loading, growth-safe heap views); marshaling proven against a JS reference ABI; `build-wasm.sh` recipe + browser demo |

The Java example builds its model in-process with SameDiff and verifies
against a live reference; the non-JVM examples run without any JVM in the
process — they load the shared `models/mlp.sdz` fixture (regenerate it with
the Java project's `GenerateExampleModel` tool) and verify against the baked
canonical vector documented in `models/README.md`. Each directory's README
covers its API surface, the conventions it follows (with sources), and
runtime-library resolution (`SDX_RUNTIME_HOME` / `SDX_RUNTIME_LIBRARY_DIR` /
linker paths) for unpacked SDK packages and the JVM-free `libsdx_*`
standalone runtime.

For the language binding source code (wrapper libraries), see
`libnd4j/include/dsp/runtime/bindings/` in the main
[deeplearning4j](https://github.com/eclipse/deeplearning4j) repository.

## LLM / VLM / STT examples (AOT package, `sdx_llm_c.h`)

The SDX AOT package (`sdx-aot-<version>-<platform>-<variant>-aot.zip`,
ADR 0109 in the main repository) adds a second C ABI on top of graph
execution: `lib/libsdx_llm.so` exposes the full Java LLM stack with **no JVM**
— GGUF loading, tokenization, autoregressive generation
(`sdxLlm*`), SmolDocling document extraction (`sdxVlmExtract`) and Whisper
speech-to-text (`sdxAudioTranscribe`) — plus the `bin/sdx-llm` CLI
(`generate` / `import` GGUF→SDZ / `tokenize` / `info` / `vlm` / `transcribe`).

Each end-to-end project now also ships an idiomatic wrapper for that ABI and
a runnable LLM example (tokenize round-trip, greedy generation with a
canonical-text check, generation stats DTO, optional `--vlm`/`--transcribe`):

| Language | LLM entry point | Run |
|----------|-----------------|-----|
| Java | `LlmEndToEnd` + `SdxLlmEnvironment`/`SdxLlmModel` (JNA) | `mvn -q compile exec:java -Dexec.mainClass=org.nd4j.examples.sdx.LlmEndToEnd` |
| Kotlin | `LlmEndToEnd.kt` + `SdxLlmRuntime.kt` (JNA, data classes, `use {}`) | `gradle llmRun` |
| Python | `llm_example.py` + `sdx_llm.py` (ctypes, context managers, frozen `GenerateStats`) | `python3 llm_example.py` |
| Rust | `src/bin/llm.rs` + `src/sdx_llm.rs` (`Drop` RAII, `#[non_exhaustive]` `LlmError`) | `cargo run --release --bin llm` |
| C# | `EndToEnd llm` + `SdxLlmRuntime.cs` (P/Invoke, `NativeLibrary` resolver, records) | `dotnet run -- llm` |
| Swift | `SdxLlmExample` + `SdxLlm.swift` over a `CSdxLlm` shim | `swift run SdxLlmExample` |
| TypeScript / Node.js | `src/llm_example.ts` + `src/sdx_llm.ts` (koffi) | `npm run start:llm` |

The AOT package also ships the canonical language wrappers under `wrappers/`
(same tree as the C SDK packages), including the `sdx_llm` modules these
examples use — the examples prefer `$SDX_LLM_AOT_HOME/wrappers/<language>`
and fall back to the main-repo checkout.

AOT packages come in **optimized-math spins** per platform (base, `-avx2`,
`-avx512`, `-onednn-*`, `-armcompute`, `-cudnn`, …) — the spin's tuned native
math library is what `lib/` carries, so picking the right package for your
hardware is how you opt into optimized math; the wrappers and examples are
spin-agnostic. `SDX_NATIVE_LIB_DIR` can also point at a different spin's
`lib/` explicitly.

Resolution: set `SDX_LLM_AOT_HOME` to the unpacked AOT package; wrappers load
`$SDX_LLM_AOT_HOME/lib/libsdx_llm.so`. The library side-loads its native
dependencies relative to the **host executable**, so `SDX_NATIVE_LIB_DIR`
must point at `$SDX_LLM_AOT_HOME/lib` — Python/Rust/C#/Node/Swift wrappers set
it automatically before the first load; on the JVM (Java/Kotlin) export it in
the environment before launching. A runtime handle is bound to the OS thread
that created it (one runtime per thread for concurrency).

React Native and WebAssembly are intentionally not covered: GraalVM
native-image has no Android/iOS target (on-device stays with the JVM-free C
runtime `libsdx`), and `libsdx_llm` is a native shared library rather than a
wasm module (browsers go through `serving/`) — see those READMEs.
