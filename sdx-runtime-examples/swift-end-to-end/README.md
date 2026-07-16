# SDX Runtime — Swift end-to-end example

Loads `../models/mlp.sdz` through the SDX C ABI via the `SdxRuntime` Swift
package — no JVM in the process — and walks the full SDK lifecycle using
idiomatic Swift:

1. Runtime and model creation
2. Input-contract discovery via `inputNames()`
3. Placeholder marking for the batch input
4. Warmup runs with `[String: SdxTensor]` dictionaries (no pointer juggling)
5. `freezeShapes()` → DSP replay fast path
6. Typed `SdxExecutionReport` telemetry (Swifty names, `CustomStringConvertible`)
7. Canonical output verification (≤ 1e-4)
8. Typed `SdxError` catch block

## Quick look

```swift
// Discovery — no hard-coded indices.
let names = ctx.inputNames()
// → ["w1", "b1", "w2", "b2", "x"]

// Inference — value-type tensors in, value-type tensors out.
let outputs = try ctx.run(
    inputs:       ["w1": w1, "b1": b1, "w2": w2, "b2": b2, "x": x],
    outputShapes: ["probs": [2, 3]]
)
let probs: [Float] = outputs["probs"]!.scalars

// Telemetry — no raw struct field names.
let report = try ctx.executionReport()
print(report.appliedBackend?.description ?? "?")  // "SLOT_BY_SLOT"
print(report.planPhase?.description ?? "?")        // "REPLAYING"
print(report)                                       // CustomStringConvertible
```

## Run

```bash
# The wrapper links the runtime shared library named in its module map
# (nd4jcpu by default).  Add the SDK lib dir to the linker search path:
swift run -Xlinker -L/path/to/sdk/lib SdxEndToEnd ../models/mlp.sdz
```

`Package.swift` resolves the `SdxRuntime` wrapper package from a sibling
`deeplearning4j` checkout by default.  Point the path dependency at
`<sdk>/wrappers/swift` when building against an unpacked SDK package.

Requires macOS 13+ or a Linux Swift 5.9+ toolchain (adjust the platform
guards in `Package.swift` for Linux).

---

## LLM AOT example (`SdxLlmExample`)

A second executable target in this package wraps the `sdx_llm_c.h` C ABI
(GraalVM AOT native-image — no JVM) via the `SdxLlm` Swift library.

### API at a glance

```swift
import SdxLlm

// Threading: the GraalVM isolate has single-thread affinity.
// Create, use and destroy each SdxLlmRuntime from one OS thread.
let rt = try SdxLlmRuntime()            // auto-sets SDX_NATIVE_LIB_DIR
print("ABI:", rt.abiVersion())          // must equal SDX_LLM_ABI_VERSION = 1

let model = try rt.loadModel(
    modelPath:     "~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf",
    tokenizerPath: "~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json",
    optionsJson:   #"{"maxNewTokens":8,"sampling":{"preset":"greedy"}}"#
)

let text = try model.generate("The capital of France is")
print(text)   // " Paris."

if let stats = try model.lastResultStats() { print(stats) }

// Tokenize / detokenize
let ids = try model.tokenize("hello world", addSpecialTokens: false)
let rt2 = try model.detokenize(ids, skipSpecialTokens: true)

// VLM (SmolDocling) and Whisper — stateless, one call each:
let extracted  = try rt.vlmExtract(modelPath: vmPath, inputPath: img,
                                   optionsJson: #"{"format":"markdown"}"#)
let transcript = try rt.audioTranscribe(modelPath: wsPath, audioPath: wav)

model.close(); rt.close()   // also called automatically on deinit
```

### Error handling

```swift
do {
    let model = try rt.loadModel(modelPath: "/bad/path.gguf")
} catch let e as SdxLlmError {
    print(e)   // SdxLlmError(MODEL_LOAD_FAILED): …
}
```

Status codes map to `SdxLlmStatus` enum cases:
`ok` / `invalidArgument` / `modelLoadFailed` / `executionFailed` / `ioError`.

### Run

```bash
export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8

swift run \
  -Xcc -I$SDX_LLM_AOT_HOME/include \
  -Xlinker -L$SDX_LLM_AOT_HOME/lib \
  SdxLlmExample \
  ~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
  ~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json
```

`SdxLlmRuntime()` automatically sets `SDX_NATIVE_LIB_DIR` from `SDX_LLM_AOT_HOME`
when `SDX_NATIVE_LIB_DIR` is not already in the environment.  This ensures that
`libsdx_llm` can find its side-loaded native companions (ND4J CPU kernels,
tokenizers) even when the Swift binary and the SDK `lib/` directory are not
co-located.

CPU generation with Qwen3.5-0.8B takes ~1–3 minutes on first run (GGUF import +
plan compilation).  Subsequent calls on the same model object reuse the compiled
plan and KV-cache state.

Requires macOS 13+ or Linux Swift 5.9+.  No JVM needed at runtime.

### Deployment layout

```
$SDX_LLM_AOT_HOME/
  include/   sdx_llm_c.h (pass -Xcc -I here)
  lib/       libsdx_llm.so  libjnind4jcpu.so  libtokenizers_wrapper.so  …
  bin/       sdx-llm  (CLI)
```

---

## Fixture

`../models/mlp.sdz` is generated once by the Java `GenerateExampleModel` tool
in the `java-end-to-end` sibling example.  The fixture implements a two-layer
MLP with:

| Input | Shape  | Description          |
|-------|--------|----------------------|
| `w1`  | [4, 8] | First-layer weight   |
| `b1`  | [8]    | First-layer bias     |
| `w2`  | [8, 3] | Second-layer weight  |
| `b2`  | [3]    | Second-layer bias    |
| `x`   | [2, 4] | Batch input          |

Output `probs` has shape `[2, 3]` (batch=2, classes=3, softmax-normalised).

Canonical verification: `x = [0.1 … 0.8]` → `probs ≈ [0.4448, 0.3220, 0.2331, 0.4567, 0.3196, 0.2237]`.
