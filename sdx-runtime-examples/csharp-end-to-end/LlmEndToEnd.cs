// SPDX-License-Identifier: Apache-2.0
//
// LlmEndToEnd.cs — SDX LLM C ABI end-to-end example for C#.
//
// Demonstrates the full LLM lifecycle following OnnxRuntime-C# conventions:
//   • SdxLlmRuntime / SdxLlmModel as IDisposable, using-declarations
//   • record LlmResultStats — value equality, auto-ToString
//   • NativeLibrary resolver honoring SDX_LLM_AOT_HOME
//   • Environment.SetEnvironmentVariable for SDX_NATIVE_LIB_DIR (unlike JVM)
//   • Greedy generation asserted to contain "Paris"
//
// Run:
//   export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
//   dotnet run -- llm [path/to/model.gguf] [path/to/tokenizer.json]

#nullable enable

using System;
using System.IO;
using Nd4j.Dsp.Runtime.Llm;

namespace Nd4j.Examples.Sdx;

/// <summary>
/// End-to-end walkthrough of the SDX LLM C ABI from C# — no JVM in the process.
///
/// <para>
/// This example embeds <c>libsdx_llm.so</c>, the AOT-compiled (GraalVM
/// native-image) LLM library, from a pure .NET process via P/Invoke.  The
/// entire Java LLM stack (GGUF import, HuggingFace tokenization, KV-cache,
/// autoregressive generation) was compiled ahead-of-time into a plain C ABI.
/// The POINT of this example is showing that a .NET host can embed the AOT
/// library without any Java runtime installed.
/// </para>
///
/// <para>
/// Unlike the JVM wrappers, <c>SdxLlmRuntime.Create()</c> can set
/// <c>SDX_NATIVE_LIB_DIR</c> in the process environment before the first
/// P/Invoke — <see cref="Environment.SetEnvironmentVariable"/> propagates to
/// <c>getenv()</c> in native code on Linux.
/// </para>
///
/// <para>Steps demonstrated:</para>
/// <list type="number">
///   <item>Create <see cref="SdxLlmRuntime"/> + ABI version check.</item>
///   <item>Load a GGUF model via <see cref="SdxLlmRuntime.LoadModel"/>.</item>
///   <item>Query model info JSON.</item>
///   <item>Tokenize / detokenize round-trip.</item>
///   <item>Greedy generation — assert output contains "Paris".</item>
///   <item>Parse <see cref="LlmResultStats"/> record.</item>
///   <item>Error handling — bogus path throws <see cref="InvalidOperationException"/>.</item>
/// </list>
/// </summary>
public static class LlmEndToEnd
{
    private static readonly string DefaultModelPath =
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "dl4j-llm-models", "Qwen3.5-0.8B-Q4_K_M.gguf");

    private static readonly string DefaultTokenizerPath =
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "dl4j-llm-models", "qwen35-0.8B-tokenizer.json");

    private const string ProbePrompt       = "The capital of France is";
    private const string ExpectedSubstring = "Paris";
    private const string Greedy8Options    =
        "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}";

    /// <summary>Entry point for the LLM example.</summary>
    /// <param name="args">
    /// Optional: <c>args[0]</c> = model path, <c>args[1]</c> = tokenizer path.
    /// </param>
    /// <returns>0 on success, non-zero on failure.</returns>
    public static int Run(string[] args)
    {
        var modelPath     = args.Length > 0 ? args[0] : DefaultModelPath;
        var tokenizerPath = args.Length > 1 ? args[1] : DefaultTokenizerPath;

        Console.WriteLine("=== SDX LLM C# end-to-end ===");
        Console.WriteLine($"Model    : {modelPath}");
        Console.WriteLine($"Tokenizer: {tokenizerPath}");
        Console.WriteLine();

        // ── Step 1: create runtime ────────────────────────────────────────────
        Console.WriteLine("== Step 1: create SdxLlmRuntime (analogous to OrtEnvironment) ==");

        // C# CAN set SDX_NATIVE_LIB_DIR here — unlike JVM wrappers.
        // SdxLlmRuntime.Create() does this automatically when SDX_LLM_AOT_HOME is set.
        using var runtime = SdxLlmRuntime.Create();
        var abiVer = runtime.AbiVersion();
        Console.WriteLine($"ABI version: {abiVer} (expected {SdxLlmConstants.SDX_LLM_ABI_VERSION})");
        if (abiVer != SdxLlmConstants.SDX_LLM_ABI_VERSION)
            Console.Error.WriteLine("WARNING: ABI version mismatch — wrapper may be out of date.");
        Console.WriteLine();

        // ── Step 2: load model ────────────────────────────────────────────────
        Console.WriteLine("== Step 2: LoadModel (first load compiles DSP plan, 1-3 min on CPU) ==");
        Console.WriteLine("Loading model…");

        using var model = runtime.LoadModel(modelPath, tokenizerPath);
        Console.WriteLine("Model loaded successfully.");
        Console.WriteLine();

        // ── Step 3: model info ────────────────────────────────────────────────
        Console.WriteLine("== Step 3: model info JSON ==");
        var info = model.InfoJson();
        Console.WriteLine(Truncate(info, 400));
        Console.WriteLine();

        // ── Step 4: tokenize / detokenize ─────────────────────────────────────
        Console.WriteLine("== Step 4: tokenize / detokenize round-trip ==");
        var ids = model.Tokenize(ProbePrompt, addSpecialTokens: false);
        Console.WriteLine($"Tokenize(\"{ProbePrompt}\") → {ids.Length} tokens: [{TruncIds(ids)}]");
        var back = model.Detokenize(ids, skipSpecialTokens: true);
        Console.WriteLine($"Detokenize → \"{back}\"");
        Console.WriteLine();

        // ── Step 5: greedy generation ──────────────────────────────────────────
        Console.WriteLine("== Step 5: generate (greedy, 8 tokens) ==");
        Console.WriteLine($"Prompt: \"{ProbePrompt}\"");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var generated = model.Generate(ProbePrompt, Greedy8Options);
        sw.Stop();
        Console.WriteLine($"Output : \"{generated}\"");
        Console.WriteLine($"Elapsed: {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        var containsParis = generated.Contains(ExpectedSubstring,
            StringComparison.OrdinalIgnoreCase);
        Console.WriteLine($"Contains \"{ExpectedSubstring}\": {(containsParis ? "YES ✓" : "NO ✗")}");
        Console.WriteLine();

        // ── Step 6: result stats (record DTO) ────────────────────────────────
        Console.WriteLine("== Step 6: last result stats (record DTO) ==");
        var stats = model.LastResultStats();
        // record compiler synthesises ToString:
        Console.WriteLine($"  {stats}");
        if (stats.TokensPerSec.HasValue)
            Console.WriteLine($"  tokensPerSec = {stats.TokensPerSec:F2}");
        Console.WriteLine();

        // ── Step 7: error handling ─────────────────────────────────────────────
        Console.WriteLine("== Step 7: error handling ==");
        try
        {
            runtime.LoadModel("/definitely/not/a/model.gguf");
            Console.Error.WriteLine("UNEXPECTED: bogus load succeeded.");
            return 1;
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Loading a bogus path raised: {Truncate(ex.Message, 120)}");
        }
        Console.WriteLine();

        if (!containsParis)
        {
            Console.Error.WriteLine(
                $"FAILURE: generated text does not contain \"{ExpectedSubstring}\".");
            return 1;
        }
        Console.WriteLine(
            "SUCCESS: SDX LLM C ABI verified from C# (.NET, no JVM, no samediff-llm).");
        return 0;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string Truncate(string s, int maxLen) =>
        s.Length <= maxLen ? s : s[..maxLen] + $" …[{s.Length} chars]";

    private static string TruncIds(int[] ids)
    {
        if (ids.Length == 0) return string.Empty;
        var n = Math.Min(ids.Length, 8);
        var sb = new System.Text.StringBuilder();
        for (var i = 0; i < n; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(ids[i]);
        }
        if (ids.Length > 8) sb.Append(", …");
        return sb.ToString();
    }
}
