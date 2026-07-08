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
├── SdxRuntimeEndToEndExample.java   — main walkthrough (reads like ORT client code)
├── GenerateExampleModel.java        — builds models/mlp.sdz for non-JVM examples
└── client/                          — reusable client library
    ├── SdxEnvironment.java          — runtime + session factory (AutoCloseable)
    ├── SdxSession.java              — inference session, run() + freezeShapes()
    ├── SdxTensor.java               — float32 tensor with shape
    ├── SdxSessionOptions.java       — backend / GPU / JIT configuration builder
    ├── SdxExecutionReport.java      — execution telemetry POJO
    └── SdxAbi.java                  — package-private JNA binding (dsp_runtime_c.h)
```
