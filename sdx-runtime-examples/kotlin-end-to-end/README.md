# SDX Runtime — Kotlin end-to-end example

A showcase of idiomatic Kotlin against the SDX C runtime ABI.

Loads `../models/mlp.sdz` through the [KotlinSdxRuntime] Kotlin facade (backed
by the JNA Java wrapper) — **no ND4J on the classpath** — and walks the full
SDK lifecycle:

| Step | What it demonstrates |
|------|----------------------|
| 1 | Create runtime, load model, open context |
| 2 | Discover the plan's input contract via `inputNames()` |
| 3 | Mark placeholders with `markPlaceholders("x")` |
| 4 | Warmup runs (`SLOT_BY_SLOT` phase) |
| 5 | `freezeShapes()` → DSP fast-path replay (`REPLAYING` phase) |
| 6 | Typed `ExecutionReport` data class with `.summary()` |
| 7 | Canonical output verification (≤ 1 × 10⁻⁴) |
| 8 | Error handling via `runCatching {}` |

## Kotlin idioms used

* `AutoCloseable.use {}` chains — no manual `try/finally`
* `data class ExecutionReport` with camelCase fields, `summary()`, and
  `executionTimeMs` computed property
* `enum class SdxBackend` / `PlanPhase` — typed alternatives to bare `Int` codes
* `class FloatTensor` — wraps JNA `Memory` + `TensorView`, exposes `readBack()`
* `ctx.runNamed(inputs = mapOf(...), weights = ..., outputs = ...)` — eliminates
  positional array juggling
* `ctx.markPlaceholders("x")` — bulk placeholder marking by name
* Extension properties `isReplaying`, `phaseLabel`
* `runCatching {}` for the error path
* Named and default arguments throughout
* `require()` / `check()` preconditions instead of verbose `if`/`exitProcess`

## Run

```bash
# Auto-discovers sdx_cpu / sdx_cuda / nd4jcpu from the JNA library path.
# The wrapper also checks for binding.json up to 6 parent directories.
gradle run --args="../models/mlp.sdz"

# Explicit SDK lib directory:
gradle run --args="../models/mlp.sdz" \
  -Dorg.gradle.jvmargs="-Djna.library.path=/path/to/sdk/lib"

# Sibling checkout layout (default):
#   ../deeplearning4j/libnd4j/include/dsp/runtime/bindings/  ← Kotlin + Java wrappers
#   ../deeplearning4j/nd4j/.../nd4j-sdx/src/main/java/       ← Java wrapper fallback
#
# SDK package layout override:
gradle run -PsdxWrappersDir=/path/to/sdk/wrappers --args="../models/mlp.sdz"
```

## Key APIs (Kotlin facade)

```kotlin
// Full lifecycle with use{} resource management
KotlinSdxRuntime.create().use { runtime ->
    runtime.loadModel("model.sdz").use { model ->
        model.createContext(outputs = listOf("probs")).use { ctx ->

            // Input contract discovery
            val names: List<String> = ctx.inputNames()

            // Bulk placeholder marking by name
            ctx.markPlaceholders("x")

            // Named-input run — order resolved automatically
            val output = FloatTensor.zeros(longArrayOf(2, 3))
            ctx.runNamed(
                inputs  = mapOf("x" to FloatTensor(xData, longArrayOf(2, 4))),
                weights = weightsMap,
                outputs = listOf(output),
            )
            val probs: FloatArray = output.readBack()

            // Freeze → replay fast path
            ctx.freezeShapes()
            println(ctx.phaseLabel)      // "REPLAYING"
            println(ctx.isReplaying)     // true

            // Typed execution report
            val report: ExecutionReport = ctx.executionReport()
            println(report.summary())
            println(report.executionTimeMs)   // ms as Double
        }
    }
}
```

## Model contract (`models/mlp.sdz`)

| Input name | Shape   | dtype  | Role        |
|------------|---------|--------|-------------|
| `w1`       | `[4,8]` | float32 | weight     |
| `b1`       | `[8]`   | float32 | bias       |
| `w2`       | `[8,3]` | float32 | weight     |
| `b2`       | `[3]`   | float32 | bias       |
| `x`        | `[B,4]` | float32 | placeholder |

Output `probs[B,3]` — row-wise softmax. For `x = [0.1 .. 0.8]` (2 rows):

```
row 0: [0.44481823, 0.32203630, 0.23314552]
row 1: [0.45671480, 0.31961477, 0.22367041]
```
