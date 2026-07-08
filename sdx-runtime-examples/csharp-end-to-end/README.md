# SDX Runtime — C# end-to-end example

Loads `../models/mlp.sdz` through the SDX C ABI via the `SdxRuntime` P/Invoke
wrapper — no JVM in the process — and walks the full SDK lifecycle following
[OnnxRuntime-C# conventions](https://onnxruntime.ai/docs/get-started/with-csharp.html).

## What this example demonstrates

| Concept | Implementation |
|---|---|
| Named-input `Run()` | `IReadOnlyDictionary<string, DenseTensor<float>>` mapped to positional binding via `InputNames()` |
| Zero-copy tensor interop | `GCHandle.Alloc(arr, GCHandleType.Pinned)` + `IntPtr.Add` for offset; pin held only for the `Run()` call |
| Idiomatic resource management | `using var` declarations (C# 8 / net6), nested disposal in reverse declaration order |
| Result DTO | `record ExecutionReport(...)` — value equality and auto-`ToString` from the compiler |
| Tensor type | `DenseTensor<float>` from `Microsoft.ML.OnnxRuntime.Managed` (managed-only, no native ORT DLLs) in the example layer; the SDK wrapper (`SdxRuntime.cs`) stays dependency-free |
| Nullable safety | `#nullable enable` throughout |
| Input-contract discovery | `ctx.InputNames()` → build dictionary; no hardcoded indices |

## Architecture

```
EndToEnd.cs  (example layer)
  └─ SdxSession           — named-input Run(), ExecutionReport DTO
  └─ ExecutionReport      — record DTO (positional, immutable, value equality)
  └─ EndToEnd.Main(...)   — lifecycle walkthrough

SdxRuntime.cs  (SDK wrapper — no dependencies, included via Compile Include)
  └─ SdxRuntime           — create/destroy runtime, load models
  └─ SdxModel             — load/unload model bundles
  └─ SdxContext           — execute, freeze shapes, query plan state
  └─ SdxTensorViewLease   — manage unmanaged shape allocation for SdxTensorView
```

## Run

```bash
# The wrapper probes SDK-relative lib/ locations and default library names;
# pass an explicit runtime path as the second argument if needed.
dotnet run -- [path/to/model.sdz] [path/to/libsdx_cpu.so]
```

The project compiles the wrapper source (`SdxRuntime.cs`) from a sibling
`deeplearning4j` checkout by default; override with
`dotnet run -p:SdxWrapperDir=/path/to/sdk/wrappers/csharp`.

## Expected output

```
== Step 1: create runtime and load mlp.sdz ==
SDX runtime ABI version: 1

== Step 2: discover input contract ==
Plan expects 5 external inputs, 1 output(s):
  input[0] = "w1"
  input[1] = "b1"
  input[2] = "w2"
  input[3] = "b2"
  input[4] = "x"

== Step 3: warmup runs ==
  run 1: rows sum to 1: True; matches canonical expectation: True (maxDiff=...)
  run 2: rows sum to 1: True
  run 3: rows sum to 1: True

== Step 4: FreezeShapes() → DSP replay fast path ==
Plan phase after freeze: 1 (SHAPES_FROZEN)
  run 4: rows sum to 1: True
  ...

== Step 5: execution report ==
  ExecutionReport { AppliedBackend = AUTO, PlanPhase = REPLAYING, UsedFallback = False, ... }

== Step 6: error handling ==
Loading a bogus path raised: sdxLoadBundle failed: ...

SUCCESS: SDX C ABI outputs verified from pure C# (no JVM).
```

## Conventions followed

- [.NET Framework Design Guidelines — Dispose Pattern](https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/dispose-pattern)
- [Microsoft.ML.OnnxRuntime C# API reference](https://onnxruntime.ai/docs/api/csharp-api.html) (named-input pattern, DenseTensor, using-declarations)
- [Memory\<T\> / Span\<T\> usage guidelines — Rule 9 (P/Invoke pinning)](https://learn.microsoft.com/en-us/dotnet/standard/memory-and-spans/memory-t-usage-guidelines)
- [C# record reference — DTOs](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/builtin-types/record)
