/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ******************************************************************************
 */
package org.nd4j.examples.sdx

import kotlin.system.exitProcess

/**
 * End-to-end walkthrough of the SDX LLM C ABI (`sdx_llm_c.h`) from Kotlin.
 *
 * This example embeds the **JVM-free** AOT-compiled LLM library
 * (`libsdx_llm.so`) from a Kotlin/JVM process via JNA — no `samediff-llm`,
 * no ND4J, no JVM spinup in the library. GraalVM native-image compiled the
 * whole Java LLM stack into a plain C ABI; this example is the consumer.
 *
 * ## What this example demonstrates
 *
 * | Step | What |
 * |------|------|
 * | 1 | Create [SdxLlmRuntime] + ABI version check |
 * | 2 | Load a GGUF model — DSP plan compiled on first load |
 * | 3 | Query model info JSON |
 * | 4 | Tokenize / detokenize round-trip |
 * | 5 | Greedy generation — assert output contains "Paris" |
 * | 6 | Parse stats via [LlmResultStats] data class |
 * | 7 | Error handling via `runCatching {}` |
 *
 * ## Prerequisites
 *
 * ```bash
 * export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
 * export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 * ```
 * `SDX_NATIVE_LIB_DIR` **must** be set before the JVM starts — a JVM process
 * cannot set process environment after launch, and `libsdx_llm.so` resolves
 * side-loaded natives relative to the host executable via this variable.
 *
 * ## Run
 *
 * ```bash
 * export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
 * export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 * gradle llmRun --args="${HOME}/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
 *                        ${HOME}/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json"
 * ```
 */

private val DEFAULT_MODEL_PATH =
    "${System.getProperty("user.home")}/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf"
private val DEFAULT_TOKENIZER_PATH =
    "${System.getProperty("user.home")}/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json"

private const val PROBE_PROMPT       = "The capital of France is"
private const val EXPECTED_SUBSTRING = "Paris"
private const val GREEDY_8_OPTIONS   =
    """{"maxNewTokens":8,"sampling":{"preset":"greedy"}}"""

fun main(args: Array<String>) {
    val modelPath     = args.getOrElse(0) { DEFAULT_MODEL_PATH }
    val tokenizerPath = args.getOrElse(1) { DEFAULT_TOKENIZER_PATH }

    println("=== SDX LLM Kotlin end-to-end ===")
    println("Model    : $modelPath")
    println("Tokenizer: $tokenizerPath")
    println()

    // ── Step 1: create runtime ────────────────────────────────────────────────
    println("Step 1 — create SdxLlmRuntime (binds libsdx_llm.so via JNA)")
    SdxLlmRuntime.create().use { runtime ->
        val abiVer = runtime.abiVersion()
        println("  ABI version: $abiVer (expected ${SdxLlmRuntime.SDX_LLM_ABI_VERSION})")
        check(abiVer == SdxLlmRuntime.SDX_LLM_ABI_VERSION) {
            "ABI version mismatch: got $abiVer, expected ${SdxLlmRuntime.SDX_LLM_ABI_VERSION}"
        }
        println()

        // ── Step 2: load model ────────────────────────────────────────────────
        println("Step 2 — loadModel (first load compiles DSP plan, 1–3 min on CPU)")
        runtime.loadModel(
            modelPath     = modelPath,
            tokenizerPath = tokenizerPath,
        ).use { model ->
            println("  Model loaded successfully.")
            println()

            // ── Step 3: model info ────────────────────────────────────────────
            println("Step 3 — model info JSON")
            val info = model.infoJson()
            println("  " + info.take(300) + if (info.length > 300) " …[${info.length} chars]" else "")
            println()

            // ── Step 4: tokenize / detokenize ─────────────────────────────────
            println("Step 4 — tokenize / detokenize round-trip")
            val ids = model.tokenize(PROBE_PROMPT, addSpecialTokens = false)
            println("  tokenize(\"$PROBE_PROMPT\") → ${ids.size} tokens: ${ids.take(8)}…")
            val back = model.detokenize(ids, skipSpecialTokens = true)
            println("  detokenize → \"$back\"")
            println()

            // ── Step 5: greedy generation ──────────────────────────────────────
            println("Step 5 — generate (greedy, 8 tokens)")
            println("  Prompt: \"$PROBE_PROMPT\"")
            val t0 = System.currentTimeMillis()
            val generated = model.generate(PROBE_PROMPT, GREEDY_8_OPTIONS)
            val ms = System.currentTimeMillis() - t0
            println("  Output : \"$generated\"")
            println("  Elapsed: ${ms} ms")
            println()

            val containsParis = generated.contains(EXPECTED_SUBSTRING)
            println("  Contains \"$EXPECTED_SUBSTRING\": ${if (containsParis) "YES ✓" else "NO ✗"}")
            println()

            // ── Step 6: generation stats ──────────────────────────────────────
            println("Step 6 — generation stats")
            val stats = model.lastResultStats()
            println(stats.summary().lines().joinToString("\n") { "  $it" }.trimEnd())
            println()

            // ── Step 7: error handling ────────────────────────────────────────
            println("Step 7 — error handling (bogus model path)")
            runCatching {
                runtime.loadModel("/definitely/not/a/model.gguf")
            }.onSuccess {
                System.err.println("UNEXPECTED: bogus load succeeded")
                exitProcess(1)
            }.onFailure { e ->
                println("  Caught ${e::class.simpleName}: ${e.message?.take(120)}")
            }
            println()

            require(containsParis) {
                "FAILURE: generated text does not contain \"$EXPECTED_SUBSTRING\"."
            }
        }
    }

    println("SUCCESS — SDX LLM C ABI verified from Kotlin (no samediff-llm on classpath).")
}

private fun IntArray.take(n: Int) =
    toList().take(n).let { if (size > n) "$it…" else it.toString() }
