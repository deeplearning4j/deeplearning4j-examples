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

---

## LLM example (`LlmEndToEnd.kt` / `gradle llmRun`)

Demonstrates the SDX LLM C ABI (`sdx_llm_c.h`) — the GraalVM native-image
compiled LLM library — from Kotlin via JNA.  **No `samediff-llm` or ND4J on
the classpath** — the library is JVM-free.  The POINT of this example is
embedding the AOT library from a Kotlin/JVM host.

### Kotlin LLM idioms used

* `SdxLlmRuntime.create().use { … }` — `use {}` resource chain
* `data class LlmResultStats` — camelCase fields, `summary()` method
* `enum class LlmStatus` — typed status codes instead of bare `Int`
* `model.generate(…)` / `model.tokenize(…)` / `model.detokenize(…)`
* `model.lastResultStats()` — parses JSON into typed `LlmResultStats`
* `runCatching {}` for the error path
* Top-level `vlmExtract()` / `audioTranscribe()` helpers

### Threading

The runtime handle is bound to the OS thread that created it.  Create, use, and
destroy from **one thread**.

### Side-loaded natives (CRITICAL for JVM)

`libsdx_llm.so` resolves its side-loaded natives relative to the host executable
via `SDX_NATIVE_LIB_DIR`. A JVM **cannot** set process environment after start.
Export before launching Gradle:

```bash
export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
```

### Run

```bash
export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib

gradle llmRun --args="${HOME}/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
                       ${HOME}/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json"
```

### Key LLM APIs

```kotlin
SdxLlmRuntime.create().use { runtime ->
    runtime.loadModel(modelPath, tokenizerPath).use { model ->

        // Greedy generation
        val text = model.generate(
            prompt = "The capital of France is",
            optionsJson = """{"maxNewTokens":8,"sampling":{"preset":"greedy"}}""",
        )
        println(text)  // " Paris."

        // Typed stats
        val stats: LlmResultStats = model.lastResultStats()
        println(stats.summary())

        // Tokenization round-trip
        val ids: IntArray = model.tokenize("Hello world")
        val back: String  = model.detokenize(ids)
    }
}
```

---

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
