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
