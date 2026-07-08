// SPDX-License-Identifier: Apache-2.0
//
// SDX Runtime — idiomatic .NET ML client example.
//
// Demonstrates the full SDK lifecycle following OnnxRuntime-C# conventions:
//   • Named-input Run() via IReadOnlyDictionary<string, DenseTensor<float>>
//   • Zero-copy GCHandle pinning for the duration of each Run() call only
//   • using-declarations (C# 8+ / net6) and #nullable enable throughout
//   • record ExecutionReport DTO with value-equality and auto-ToString
//   • DenseTensor<float> (Microsoft.ML.OnnxRuntime.Managed, tensors-only, no
//     native ORT inference DLLs) in the example layer — SdxRuntime.cs wrapper
//     stays dependency-free
//
// Run:
//   dotnet run -- [path/to/model.sdz] [path/to/libsdx_cpu.so]

#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nd4j.Dsp.Runtime;

namespace Nd4j.Examples.Sdx;

// ---------------------------------------------------------------------------
// Example-layer abstractions (OnnxRuntime-C# idioms adapted for SDX)
// ---------------------------------------------------------------------------

/// <summary>
/// Immutable telemetry snapshot for a completed inference run.
/// Value equality and auto-formatted <c>ToString</c> are synthesised by the
/// record compiler — no boilerplate needed.
/// </summary>
/// <param name="AppliedBackend">Backend name that was actually used (e.g. "AUTO", "TRITON").</param>
/// <param name="PlanPhase">DSP plan phase name after execution (e.g. "REPLAYING").</param>
/// <param name="UsedFallback">Whether the runtime fell back from the requested path.</param>
/// <param name="ExecutionCount">Total number of Run() calls completed on the context.</param>
/// <param name="ExecutionTimeMs">Wall-clock duration of the last Run() call in milliseconds.</param>
public record ExecutionReport(
    string AppliedBackend,
    string PlanPhase,
    bool UsedFallback,
    int ExecutionCount,
    double ExecutionTimeMs);

/// <summary>
/// Thin convenience layer over <see cref="SdxContext"/> that maps named
/// <see cref="DenseTensor{T}"/> inputs to the positional binding contract
/// expected by the SDX C ABI, mirroring the OnnxRuntime-C# pattern:
/// <c>session.Run(new Dictionary&lt;string, OrtValue&gt; { { "input", value } }, ...)</c>.
/// </summary>
/// <remarks>
/// The session does NOT take ownership of the underlying <see cref="SdxContext"/>
/// — the caller is responsible for its lifetime.
/// </remarks>
public sealed class SdxSession
{
    private static readonly string[] BackendNames =
        { "AUTO", "SLOT_BY_SLOT", "CUDA_GRAPHS", "NVRTC", "PTX", "TRITON", "MLX", "ARM_HYBRID", "NNAPI" };

    private static readonly string[] PhaseNames =
        { "SLOT_BY_SLOT (warmup)", "SHAPES_FROZEN", "REPLAYING", "REPLAY_BLOCKED" };

    private readonly SdxContext _ctx;

    /// <param name="context">The execution context to delegate to.</param>
    public SdxSession(SdxContext context) => _ctx = context;

    /// <summary>
    /// External input names in plan binding order.  Use these as the keys for
    /// the <paramref name="inputs"/> dictionary passed to <see cref="Run"/>.
    /// </summary>
    public string?[] InputNames() => _ctx.InputNames();

    /// <summary>
    /// Executes the plan with named float32 input tensors and pre-allocated
    /// float32 output tensors.
    /// </summary>
    /// <param name="inputs">
    /// Map from plan input name to tensor.  Every name returned by
    /// <see cref="InputNames"/> must be present as a key.
    /// </param>
    /// <param name="outputs">
    /// Pre-allocated output tensors in plan output order.  The plan writes
    /// results in-place; element counts must match the plan's output shapes.
    /// </param>
    /// <param name="options">Per-call options; <see langword="null"/> uses defaults.</param>
    /// <exception cref="KeyNotFoundException">
    /// Thrown when a required plan input name is absent from <paramref name="inputs"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a tensor's backing buffer cannot be pinned.
    /// </exception>
    public void Run(
        IReadOnlyDictionary<string, DenseTensor<float>> inputs,
        DenseTensor<float>[] outputs,
        SdxRunOptions? options = null)
    {
        var planInputNames = _ctx.InputNames();

        var inputViews   = new SdxTensorView[planInputNames.Length];
        var outputViews  = new SdxTensorView[outputs.Length];

        // Nullable arrays: elements start null; the finally block uses ?. to skip un-initialised slots.
        var inputLeases  = new SdxTensorViewLease?[planInputNames.Length];
        var outputLeases = new SdxTensorViewLease?[outputs.Length];
        var inputPins    = new GCHandle[planInputNames.Length];
        var outputPins   = new GCHandle[outputs.Length];

        try
        {
            // Build input views in positional plan order — pin only for this Run() call.
            for (var i = 0; i < planInputNames.Length; i++)
            {
                var name = planInputNames[i]
                    ?? throw new InvalidOperationException(
                        $"Plan input at index {i} has a null name.");
                if (!inputs.TryGetValue(name, out var tensor))
                    throw new KeyNotFoundException(
                        $"Input '{name}' (plan index {i}) is missing from the inputs dictionary.");

                (inputPins[i], inputLeases[i]) = PinTensor(tensor);
                inputViews[i] = inputLeases[i]!.View;
            }

            // Build output views — caller pre-allocates; plan writes in-place.
            for (var i = 0; i < outputs.Length; i++)
            {
                (outputPins[i], outputLeases[i]) = PinTensor(outputs[i]);
                outputViews[i] = outputLeases[i]!.View;
            }

            _ctx.Run(inputViews, outputViews, options);
        }
        finally
        {
            // Release in LIFO order — outputs last created, released first.
            for (var i = outputs.Length - 1; i >= 0; i--)
            {
                outputLeases[i]?.Dispose();
                if (outputPins[i].IsAllocated) outputPins[i].Free();
            }
            for (var i = planInputNames.Length - 1; i >= 0; i--)
            {
                inputLeases[i]?.Dispose();
                if (inputPins[i].IsAllocated) inputPins[i].Free();
            }
        }
    }

    /// <summary>Returns an <see cref="ExecutionReport"/> for the most recent Run() call.</summary>
    public ExecutionReport GetExecutionReport()
    {
        var raw = _ctx.ExecutionReport();
        return new ExecutionReport(
            AppliedBackend:  NameOf(BackendNames, raw.applied_backend),
            PlanPhase:       NameOf(PhaseNames, raw.plan_phase),
            UsedFallback:    raw.used_fallback == 1,
            ExecutionCount:  raw.execution_count,
            ExecutionTimeMs: raw.execution_time_ns / 1_000_000.0);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// <summary>
    /// Extracts the backing <c>float[]</c> from a <see cref="DenseTensor{T}"/>,
    /// pins it with a <see cref="GCHandle"/> (preventing GC movement for the
    /// duration of the Run() call), and wraps it in an
    /// <see cref="SdxTensorViewLease"/> that owns the unmanaged shape allocation.
    /// </summary>
    /// <remarks>
    /// The caller must call <see cref="GCHandle.Free"/> and
    /// <see cref="SdxTensorViewLease.Dispose"/> after Run() returns.
    /// Both operations happen in the finally block of <see cref="Run"/>.
    /// </remarks>
    private static (GCHandle pin, SdxTensorViewLease lease) PinTensor(DenseTensor<float> tensor)
    {
        // DenseTensor<float> always backs its Memory<T> with a managed float[].
        // TryGetArray succeeds for all standard DenseTensor instances.
        if (!MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)tensor.Buffer, out ArraySegment<float> seg)
            || seg.Array is null)
        {
            throw new InvalidOperationException(
                "Cannot pin DenseTensor<float>: the backing buffer is not a managed array. " +
                "Only DenseTensor<float> instances backed by a plain float[] are supported.");
        }

        // Pin: the GC must not relocate the array while native code holds the pointer.
        var pin = GCHandle.Alloc(seg.Array, GCHandleType.Pinned);

        // Account for the segment offset (non-zero when the tensor is a slice).
        // IntPtr.Add avoids the need for unsafe pointer arithmetic.
        var dataPtr = IntPtr.Add(pin.AddrOfPinnedObject(), seg.Offset * sizeof(float));
        var bytes   = (UIntPtr)((long)tensor.Length * sizeof(float));

        // Convert int[] Dimensions to long[] shape required by SdxTensorViewLease.
        var dims = tensor.Dimensions;
        var shape = new long[dims.Length];
        for (var d = 0; d < dims.Length; d++) shape[d] = dims[d];

        var lease = SdxTensorViewLease.CreateHost(
            dataPtr, shape, dtype: 5 /* sd::DataType::FLOAT32 */, bytes);

        return (pin, lease);
    }

    private static string NameOf(string[] table, int code) =>
        code >= 0 && code < table.Length ? table[code] : $"? ({code})";
}

// ---------------------------------------------------------------------------
// Program entry point
// ---------------------------------------------------------------------------

public static class EndToEnd
{
    // Canonical verification data for models/mlp.sdz.
    // x[2,4] = [0.1..0.8]; probs[2,3] are the softmax outputs to within 1e-4.
    private static readonly float[] CanonicalX =
        { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };

    private static readonly float[] ExpectedProbs =
        { 0.44481823f, 0.3220363f, 0.23314552f, 0.4567148f, 0.31961477f, 0.22367041f };

    // ---------------------------------------------------------------------------
    // Weight factory — a real client obtains weights from the .sdz bundle or a
    // sidecar checkpoint; here we reconstruct the deterministic linspace
    // initialisers that GenerateExampleModel used to produce mlp.sdz.
    // ---------------------------------------------------------------------------

    private static DenseTensor<float> MakeWeightTensor(string name) => name switch
    {
        "w1" => MakeDense(Linspace(-1.0f,  1.0f, 32), new[] { 4, 8 }),
        "b1" => MakeDense(Linspace( 0.0f,  0.7f,  8), new[] { 8 }),
        "w2" => MakeDense(Linspace( 1.0f, -1.0f, 24), new[] { 8, 3 }),
        "b2" => MakeDense(Linspace(-0.1f,  0.1f,  3), new[] { 3 }),
        _ => throw new InvalidOperationException($"No weight value known for plan input '{name}'.")
    };

    /// <summary>
    /// Creates a <see cref="DenseTensor{T}"/> with the given shape and copies
    /// <paramref name="data"/> into its backing store.
    /// </summary>
    private static DenseTensor<float> MakeDense(float[] data, int[] shape)
    {
        var tensor = new DenseTensor<float>(shape);
        // TryGetArray always succeeds for DenseTensor<float>.
        MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)tensor.Buffer, out ArraySegment<float> seg);
        data.AsSpan().CopyTo(seg.AsSpan());
        return tensor;
    }

    private static float[] Linspace(float start, float end, int n)
    {
        var result = new float[n];
        for (var i = 0; i < n; i++)
            result[i] = start + (end - start) * i / (n - 1);
        return result;
    }

    // ---------------------------------------------------------------------------
    // Single inference step
    // ---------------------------------------------------------------------------

    private static bool RunStep(SdxSession session, string?[] inputNames, int step)
    {
        // Scale canonical x per step: step 1 uses exact canonical values so the
        // baked expectation applies; later steps change values to exercise the
        // variable-input path while keeping the shape fixed.
        var xData = new float[CanonicalX.Length];
        for (var i = 0; i < xData.Length; i++) xData[i] = CanonicalX[i] * step;

        // Build named input dictionary — weights are model constants but the SDX
        // positional contract still requires them to be supplied on every call.
        var inputs = new Dictionary<string, DenseTensor<float>>(inputNames.Length);
        foreach (var name in inputNames)
        {
            if (name is null) continue;
            inputs[name] = name == "x"
                ? MakeDense(xData, new[] { 2, 4 })
                : MakeWeightTensor(name);
        }

        // Pre-allocate the output buffer — Run() writes probs[2,3] in-place.
        var probsTensor = new DenseTensor<float>(new[] { 2, 3 });
        session.Run(inputs, new[] { probsTensor });

        // Read results from the tensor's managed backing store (zero-copy Span).
        MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)probsTensor.Buffer, out ArraySegment<float> probsSeg);
        ReadOnlySpan<float> probs = probsSeg.AsSpan();

        var row0Sum   = probs[0] + probs[1] + probs[2];
        var row1Sum   = probs[3] + probs[4] + probs[5];
        var rowSumsOk = Math.Abs(row0Sum - 1f) <= 1e-5f && Math.Abs(row1Sum - 1f) <= 1e-5f;
        var ok        = rowSumsOk;
        var checks    = $"rows sum to 1: {rowSumsOk}";

        if (step == 1)
        {
            var maxDiff = 0f;
            for (var i = 0; i < ExpectedProbs.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(probs[i] - ExpectedProbs[i]));
            var matchesCanonical = maxDiff <= 1e-4f;
            ok      = ok && matchesCanonical;
            checks += $"; matches canonical expectation: {matchesCanonical} (maxDiff={maxDiff:E2})";
        }

        Console.WriteLine($"  run {step}: {checks}");
        return ok;
    }

    // ---------------------------------------------------------------------------
    // Main
    // ---------------------------------------------------------------------------

    public static int Main(string[] args)
    {
        var modelPath = args.Length > 0
            ? args[0]
            : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/mlp.sdz"));
        if (!File.Exists(modelPath))
            modelPath = Path.GetFullPath("../models/mlp.sdz");
        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine(
                $"Model not found: {modelPath}\n" +
                "Generate it with the java-end-to-end GenerateExampleModel tool.");
            return 2;
        }

        // Step 1 — Create runtime and load model.
        Console.WriteLine($"== Step 1: create runtime and load {Path.GetFileName(modelPath)} ==");
        using var runtime = SdxRuntime.Create(args.Length > 1 ? args[1] : null);
        Console.WriteLine($"SDX runtime ABI version: {runtime.AbiVersion()}");

        // Nest model and context in using-declarations (C# 8 / net6): dispose in
        // reverse declaration order when the scope exits — identical to using(){}.
        using var model = runtime.LoadModel(modelPath);
        using var ctx   = model.CreateContext(new[] { "probs" });
        var session     = new SdxSession(ctx);

        // Step 2 — Discover input contract from the plan.
        Console.WriteLine("\n== Step 2: discover input contract ==");
        var inputNames = session.InputNames();
        Console.WriteLine($"Plan expects {ctx.NumInputs()} external inputs, {ctx.NumOutputs()} output(s):");
        for (var i = 0; i < inputNames.Length; i++)
            Console.WriteLine($"  input[{i}] = \"{inputNames[i]}\"");

        // Mark the runtime input (x) as PLACEHOLDER — shape may change between calls.
        for (var i = 0; i < inputNames.Length; i++)
            if (inputNames[i] == "x") ctx.MarkInputPlaceholder(i);

        // Step 3 — Warmup runs: let the DSP plan observe shapes and stabilise.
        Console.WriteLine("\n== Step 3: warmup runs ==");
        for (var step = 1; step <= 3; step++)
            if (!RunStep(session, inputNames, step)) return 1;

        // Step 4 — Freeze shapes: enables CUDA graph capture and the stable fast path.
        Console.WriteLine("\n== Step 4: FreezeShapes() → DSP replay fast path ==");
        ctx.FreezeShapes();
        var phaseAfterFreeze = ctx.PlanPhase();
        Console.WriteLine(
            $"Plan phase after freeze: {phaseAfterFreeze} " +
            $"({(phaseAfterFreeze == SdxConstants.SDX_PHASE_SHAPES_FROZEN ? "SHAPES_FROZEN" : "other")})");
        for (var step = 4; step <= 6; step++)
            if (!RunStep(session, inputNames, step)) return 1;

        // Step 5 — Execution report: rich telemetry via the record DTO.
        Console.WriteLine("\n== Step 5: execution report ==");
        var report = session.GetExecutionReport();
        // The record compiler synthesises ToString as:
        //   ExecutionReport { AppliedBackend = ..., PlanPhase = ..., ... }
        Console.WriteLine($"  {report}");
        Console.WriteLine($"  execution_time = {report.ExecutionTimeMs:F3} ms");

        // Step 6 — Error handling: the wrapper surfaces native errors as exceptions.
        Console.WriteLine("\n== Step 6: error handling ==");
        try
        {
            runtime.LoadModel("/definitely/not/a/model.sdz");
            Console.Error.WriteLine("UNEXPECTED: bogus load succeeded.");
            return 1;
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Loading a bogus path raised: {ex.Message}");
        }

        Console.WriteLine("\nSUCCESS: SDX C ABI outputs verified from pure C# (no JVM).");
        return 0;
    }
}
