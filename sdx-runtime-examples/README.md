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

These examples target SDX Runtime ABI version 1.

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
