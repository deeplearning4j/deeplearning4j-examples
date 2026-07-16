# SDX Runtime — Java end-to-end example

Self-contained Maven project that walks the whole SDK lifecycle against a
real model: builds a SameDiff MLP, saves it as `.sdz`, then loads and runs
it through the SDX C ABI (`dsp_runtime_c.h`) via a small reusable client
library in `org.nd4j.examples.sdx.client`.

The client API is modelled after the **ONNX Runtime Java API** — developers
familiar with `OrtEnvironment` / `OrtSession` / `OnnxTensor` will find the
patterns immediately recognisable.

## Client API summary

| SDX class | ONNX Runtime analogue | Purpose |
|---|---|---|
| `SdxEnvironment` | `OrtEnvironment` | Global runtime; factory for sessions |
| `SdxSession` | `OrtSession` | Per-model inference handle; `run()` |
| `SdxTensor` | `OnnxTensor` | Host float32 tensor with shape |
| `SdxSessionOptions` | `SessionOptions` | Backend, GPU target, JIT control |
| `SdxExecutionReport` | (profile result) | Phase, backend, timing telemetry |

The JNA plumbing (`SdxAbi`, the JNA `Structure` types) lives inside the
`client` package and is not part of the public surface — production code
would use the SDK wrapper under `wrappers/java` instead.

## Canonical usage (mirrors ONNX Runtime pattern)

```java
try (SdxEnvironment env = SdxEnvironment.create()) {
    try (SdxSession session = env.openSession("model.sdz", new String[]{"probs"})) {

        session.markInputPlaceholder("x");  // batch input changes shape

        // Named-input map — session reorders to positional slots automatically
        Map<String, SdxTensor> inputs = new LinkedHashMap<>();
        inputs.put("x",  SdxTensor.fromArray(data, new long[]{2, 4}));
        inputs.put("w1", SdxTensor.fromArray(w1,   new long[]{4, 8}));
        // ... remaining weights ...

        // Warmup (2-3 runs), then freeze shapes for the DSP replay fast path
        Map<String, SdxTensor> out = session.runWithShapes(inputs,
                Collections.singletonMap("probs", new long[]{2, 3}));
        session.freezeShapes();

        // Subsequent runs use the REPLAYING fast path
        out = session.runWithShapes(inputs, ...);
        float[] probs = out.get("probs").toFloatArray();

        SdxExecutionReport report = session.getExecutionReport();
        System.out.println(report);
    }
}
```

## Input contract

The SDX plan's external inputs cover the model's **constants**, **variables**
(weights), and **placeholders**.  Discover them at open time:

```java
session.inputNames();  // e.g. ["w1", "b1", "w2", "b2", "x"]
session.numOutputs();  // 1
```

`session.markInputPlaceholder(name)` and `session.markInputVariable(name)`
provide lifecycle hints so the DSP engine can pick the right staging strategy
per input.

## SdxTensor

```java
SdxTensor.fromArray(float[] data, long[] shape)   // copy into direct buffer
SdxTensor.fromBuffer(FloatBuffer buf, long[] shape) // zero-copy from direct buffer
tensor.update(float[] data)                         // in-place update for reuse
tensor.toFloatArray()                               // read results back
```

## Run

```bash
mvn -q compile exec:java
```

Library resolution order (first match wins):

1. System property `sdx.library` or env-var `SDX_RUNTIME_LIBRARY` — explicit path.
2. `SDX_RUNTIME_HOME/lib` from an unpacked SDK ZIP, preferring the JVM-free
   standalone runtime (`libsdx_cpu.so` / `libsdx_cuda.so`).
3. The backend library JavaCPP already extracted for this JVM process — the
   monolithic backend exports the same `sdx*` ABI.

Switch to CUDA by setting `-Dnd4j.backend=nd4j-cuda-12.9` in the pom or
overriding on the Maven command line.

## Regenerating the shared fixture

The non-JVM examples consume `../models/mlp.sdz`; regenerate it (and print
the canonical verification vector) with:

```bash
mvn -q compile exec:java -Dexec.mainClass=org.nd4j.examples.sdx.GenerateExampleModel
```

## Package layout

```
src/main/java/org/nd4j/examples/sdx/
├── SdxRuntimeEndToEndExample.java   — SameDiff graph execution walkthrough
├── LlmEndToEnd.java                 — LLM generation walkthrough (libsdx_llm.so)
├── GenerateExampleModel.java        — builds models/mlp.sdz for non-JVM examples
└── client/                          — reusable client library
    ├── SdxEnvironment.java          — runtime + session factory (AutoCloseable)
    ├── SdxSession.java              — inference session, run() + freezeShapes()
    ├── SdxTensor.java               — float32 tensor with shape
    ├── SdxSessionOptions.java       — backend / GPU / JIT configuration builder
    ├── SdxExecutionReport.java      — execution telemetry POJO
    ├── SdxAbi.java                  — package-private JNA binding (dsp_runtime_c.h)
    ├── SdxLlmEnvironment.java       — LLM runtime factory (AutoCloseable)
    ├── SdxLlmModel.java             — LLM model: generate / tokenize / info
    └── SdxLlmAbi.java               — package-private JNA binding (sdx_llm_c.h)
```

---

## LLM / VLM / STT example (`LlmEndToEnd`)

Demonstrates the SDX LLM C ABI (`sdx_llm_c.h`) — the GraalVM native-image
compiled LLM library — from Java via JNA.  **No `samediff-llm.jar` or ND4J on
the classpath** — the library is JVM-free.  The point is embedding the AOT
library from a JVM host app.

### LLM client API summary

| SDX class | ONNX Runtime analogue | Purpose |
|---|---|---|
| `SdxLlmEnvironment` | `OrtEnvironment` | GraalVM isolate + runtime; factory for models |
| `SdxLlmModel` | `InferenceSession` | Loaded model; `generate()`, `tokenize()`, `infoJson()` |

### Canonical usage

```java
try (SdxLlmEnvironment env = SdxLlmEnvironment.create()) {
    try (SdxLlmModel model = env.loadModel(modelPath, tokenizerPath, null)) {
        String text = model.generate(
            "The capital of France is",
            "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}");
        System.out.println(text);  // " Paris."
        System.out.println(model.lastResultJson());  // tok/s, finish reason, …
    }
}
```

### Threading

The runtime handle is bound to the OS thread that created it.  **Create, use,
and destroy from one thread.**  For concurrent generation, use one
`SdxLlmEnvironment` per thread — models are not shared across runtimes.

### Side-loaded natives (CRITICAL)

`libsdx_llm.so` resolves its side-loaded natives (libnd4jcpu, …) relative to
the host executable using the `SDX_NATIVE_LIB_DIR` environment variable.  A JVM
**cannot** set process environment after the JVM starts, so the runner must
export this **before** launching:

```bash
export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
```

If `SDX_LLM_AOT_HOME` is set but `SDX_NATIVE_LIB_DIR` is not, the wrapper
prints a warning at construction time.

### Run

```bash
export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib

/home/agibsonccc/dev-apps/mvn/bin/mvn -q compile exec:java \
  -Dexec.mainClass=org.nd4j.examples.sdx.LlmEndToEnd \
  -Dexec.args="$HOME/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
               $HOME/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json"
```

First run: 1–3 min (GGUF import + DSP warmup). Subsequent runs reuse the plan cache.
