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
// NOTE: This is a vendored copy kept for reference. The canonical implementation
// is org.nd4j.dsp.runtime.KotlinSdxLlmRuntime in the canonical Kotlin binding:
//   libnd4j/include/dsp/runtime/bindings/kotlin/src/main/kotlin/org/nd4j/dsp/runtime/SdxLlm.kt
// New application code should use KotlinSdxLlmRuntime directly (it is pulled in
// via the sdxWrappersDir/kotlin srcDir in build.gradle.kts).
package org.nd4j.examples.sdx

import com.sun.jna.Library
import com.sun.jna.Memory
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.ptr.IntByReference
import com.sun.jna.ptr.PointerByReference
import java.io.File
import java.nio.charset.StandardCharsets

// ─────────────────────────────────────────────────────────────────────────────
// Status codes (sdx_llm_c.h § sdx_llm_status_t)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Status codes returned by the SDX LLM C ABI.
 *
 * Values match the `sdx_llm_status_t` C enum in `sdx_llm_c.h`.
 */
enum class LlmStatus(val code: Int) {
    OK(0),
    INVALID_ARGUMENT(1),
    MODEL_LOAD_FAILED(3),
    EXECUTION_FAILED(4),
    IO_ERROR(6);

    companion object {
        fun fromCode(code: Int): LlmStatus? = entries.firstOrNull { it.code == code }
        fun label(code: Int): String = fromCode(code)?.name ?: "UNKNOWN($code)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level JNA ABI binding (package-internal)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Raw JNA binding for `sdx_llm_c.h`.  Internal — callers use [SdxLlmRuntime].
 *
 * **Threading:** the runtime handle (`sdx_llm_runtime_t*`) is bound to the
 * OS thread that created it. Create, use, and destroy from one thread.
 */
internal interface SdxLlmAbi : Library {
    fun sdxLlmCreateRuntime(): Pointer?
    fun sdxLlmDestroyRuntime(runtime: Pointer): Int
    fun sdxLlmAbiVersion(runtime: Pointer): Int

    fun sdxLlmLoadModel(
        runtime: Pointer,
        model_path: String,
        tokenizer_path: String?,
        options_json: String?,
    ): Pointer?

    fun sdxLlmUnloadModel(runtime: Pointer, model: Pointer): Int

    fun sdxLlmGenerate(
        runtime: Pointer,
        model: Pointer,
        prompt: String,
        options_json: String?,
        out_text: PointerByReference,
    ): Int

    fun sdxLlmLastResultJson(
        runtime: Pointer,
        model: Pointer,
        out_json: PointerByReference,
    ): Int

    fun sdxLlmInfoJson(
        runtime: Pointer,
        model: Pointer,
        out_json: PointerByReference,
    ): Int

    fun sdxLlmTokenize(
        runtime: Pointer,
        model: Pointer,
        text: String,
        add_special_tokens: Int,
        out_ids: PointerByReference,
        out_count: IntByReference,
    ): Int

    fun sdxLlmDetokenize(
        runtime: Pointer,
        model: Pointer,
        ids: IntArray,
        count: Int,
        skip_special_tokens: Int,
        out_text: PointerByReference,
    ): Int

    fun sdxVlmExtract(
        runtime: Pointer,
        model_path: String,
        tokenizer_path: String?,
        input_path: String,
        options_json: String?,
        out_text: PointerByReference,
    ): Int

    fun sdxAudioTranscribe(
        runtime: Pointer,
        model_path: String,
        audio_path: String,
        options_json: String?,
        out_text: PointerByReference,
    ): Int

    fun sdxLlmFree(runtime: Pointer, pointer: Pointer)

    fun sdxLlmGetLastError(runtime: Pointer, buffer: ByteArray?, capacity: Int): Int
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Stats snapshot from the most recent [SdxLlmRuntime.SdxLlmModel.generate] call.
 *
 * The raw JSON string is available via [rawJson] for fields not yet parsed here.
 *
 * @param rawJson      Full JSON string from `sdxLlmLastResultJson`.
 * @param promptTokens Number of input (prompt) tokens processed.
 * @param newTokens    Number of tokens generated.
 * @param tokensPerSec Generation throughput in tokens per second.
 * @param finishReason Why generation stopped (e.g. `"max_tokens"`, `"eos"`).
 */
data class LlmResultStats(
    val rawJson: String,
    val promptTokens: Int?,
    val newTokens: Int?,
    val tokensPerSec: Double?,
    val finishReason: String?,
) {
    fun summary(): String = buildString {
        appendLine("LlmResultStats {")
        appendLine("  promptTokens = $promptTokens")
        appendLine("  newTokens    = $newTokens")
        appendLine("  tokensPerSec = ${tokensPerSec?.let { "%.2f".format(it) } ?: "n/a"}")
        append("  finishReason = $finishReason")
        append("\n}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kotlin facade — SdxLlmRuntime
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Idiomatic Kotlin facade over the SDX LLM C ABI (`sdx_llm_c.h`).
 *
 * This wrapper binds `libsdx_llm.so` — the AOT-compiled (GraalVM native-image)
 * LLM/VLM/STT library — via JNA. **No JVM is embedded in the library.** The
 * point of this example is showing how a JVM-host app can embed the AOT library
 * without any `samediff-llm` or ND4J dependencies on its classpath.
 *
 * All three lifecycle objects ([SdxLlmRuntime], [SdxLlmModel]) implement
 * [AutoCloseable] and are designed for Kotlin's `use {}` idiom:
 *
 * ```kotlin
 * SdxLlmRuntime.create().use { runtime ->
 *     runtime.loadModel(modelPath, tokenizerPath).use { model ->
 *         val text = model.generate(
 *             prompt = "The capital of France is",
 *             optionsJson = """{"maxNewTokens":8,"sampling":{"preset":"greedy"}}""",
 *         )
 *         println(text)  // " Paris."
 *     }
 * }
 * ```
 *
 * ## Threading (v1)
 * The runtime handle is bound to the OS thread that created it.  Create, use,
 * and destroy the runtime from **one thread**.  For concurrent generation, use
 * one [SdxLlmRuntime] per thread — models are not shared across runtimes.
 *
 * ## Side-loaded natives (CRITICAL)
 * `libsdx_llm.so` resolves its side-loaded natives (libnd4jcpu, …) relative to
 * the host executable using `SDX_NATIVE_LIB_DIR`. A JVM **cannot** set process
 * environment after start. The runner must export the variable before launching:
 * ```bash
 * export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
 * export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 * gradle run --args="<model> <tokenizer>"
 * ```
 */
class SdxLlmRuntime private constructor(
    internal val api: SdxLlmAbi,
    internal val runtime: Pointer,
) : AutoCloseable {

    // ── Factory ──────────────────────────────────────────────────────────────

    companion object {
        /** ABI version this wrapper was written against. */
        const val SDX_LLM_ABI_VERSION = 1

        /**
         * Creates a runtime by auto-detecting `libsdx_llm.so`.
         *
         * Resolution order (first match wins):
         * 1. JVM property `sdx.llm.library` or env `SDX_LLM_LIBRARY`.
         * 2. `SDX_LLM_AOT_HOME/lib/libsdx_llm.so`.
         * 3. Bare names `sdx_llm` / `sdx_llm_cpu` via JNA search.
         *
         * @throws IllegalStateException if the library or runtime cannot be loaded.
         */
        fun create(): SdxLlmRuntime = create(resolveLibrary())

        /**
         * Creates a runtime loading the library at [libraryPath].
         *
         * @param libraryPath Absolute path or bare JNA library name (e.g. `"sdx_llm"`).
         */
        fun create(libraryPath: String): SdxLlmRuntime {
            warnIfNativeLibDirMissing()
            val api = Native.load(libraryPath, SdxLlmAbi::class.java)
            val rt = api.sdxLlmCreateRuntime()
                ?: throw IllegalStateException(
                    "sdxLlmCreateRuntime() returned null — check that " +
                    "SDX_NATIVE_LIB_DIR is set before starting the JVM.")
            return SdxLlmRuntime(api, rt)
        }

        // ── Library resolution ────────────────────────────────────────────────

        private fun resolveLibrary(): String {
            System.getProperty("sdx.llm.library")?.takeIf { it.isNotBlank() }?.let { return it }
            System.getenv("SDX_LLM_LIBRARY")?.takeIf { it.isNotBlank() }?.let { return it }

            val aotHome = System.getenv("SDX_LLM_AOT_HOME")
            if (!aotHome.isNullOrBlank()) {
                for (name in listOf("libsdx_llm.so", "sdx_llm.so", "libsdx_llm.dylib")) {
                    val candidate = File(aotHome, "lib/$name")
                    if (candidate.exists()) {
                        // Extend jna.library.path so side-loaded deps are found.
                        val libDir = File(aotHome, "lib").absolutePath
                        val current = System.getProperty("jna.library.path", "")
                        if (!current.contains(libDir)) {
                            System.setProperty("jna.library.path",
                                if (current.isEmpty()) libDir else "$libDir${File.pathSeparatorChar}$current")
                        }
                        return candidate.absolutePath
                    }
                }
            }
            return "sdx_llm"
        }

        private fun warnIfNativeLibDirMissing() {
            val aotHome = System.getenv("SDX_LLM_AOT_HOME")
            val nativeLibDir = System.getenv("SDX_NATIVE_LIB_DIR")
            if (!aotHome.isNullOrBlank() && nativeLibDir.isNullOrBlank()) {
                System.err.println(
                    "[SdxLlmRuntime] WARNING: SDX_LLM_AOT_HOME is set but SDX_NATIVE_LIB_DIR is not.\n" +
                    "  libsdx_llm.so resolves side-loaded natives relative to the host executable.\n" +
                    "  Export SDX_NATIVE_LIB_DIR=\$SDX_LLM_AOT_HOME/lib BEFORE starting the JVM.")
            }
        }
    }

    // ── Runtime API ──────────────────────────────────────────────────────────

    /** ABI version reported by the loaded library. */
    fun abiVersion(): Int = api.sdxLlmAbiVersion(runtime)

    /** Last error message from the native runtime, or empty string. */
    fun lastError(): String {
        val buf = ByteArray(2048)
        val len = api.sdxLlmGetLastError(runtime, buf, buf.size)
        if (len <= 0) return ""
        return String(buf, 0, minOf(len, buf.size - 1), StandardCharsets.UTF_8)
    }

    /**
     * Loads a GGUF/SDZ model and builds the generation pipeline.
     *
     * First load compiles the DSP plan (1–3 min on CPU); subsequent loads reuse
     * the plan cache.
     *
     * @param modelPath     Path to a `.gguf` / `.sdz` model file.
     * @param tokenizerPath Path to `tokenizer.json` or its directory, or `null`
     *                      to probe the model directory.
     * @param optionsJson   Optional JSON defaults, e.g.
     *                      `{"maxNewTokens":128,"sampling":{"preset":"greedy"}}`.
     * @throws IllegalStateException if loading fails.
     */
    fun loadModel(
        modelPath: String,
        tokenizerPath: String? = null,
        optionsJson: String? = null,
    ): SdxLlmModel {
        val handle = api.sdxLlmLoadModel(runtime, modelPath, tokenizerPath, optionsJson)
            ?: throw IllegalStateException(
                "sdxLlmLoadModel failed: ${lastError()}\n  model_path=$modelPath")
        return SdxLlmModel(this, handle)
    }

    override fun close() {
        api.sdxLlmDestroyRuntime(runtime)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A loaded LLM/VLM model.
     *
     * Obtain via [SdxLlmRuntime.loadModel]; always close via `use {}`.
     *
     * **Threading:** same thread as the parent [SdxLlmRuntime].
     */
    class SdxLlmModel internal constructor(
        private val rt: SdxLlmRuntime,
        private var modelHandle: Pointer,
    ) : AutoCloseable {

        // ── Generation ────────────────────────────────────────────────────────

        /**
         * Generates a text continuation for [prompt].
         *
         * The pipeline's compiled plan and KV state are reused across calls
         * (stateful decode — do NOT reset between calls on the same model).
         *
         * @param prompt      Input text (UTF-8).
         * @param optionsJson Per-call overrides, e.g.
         *                    `{"maxNewTokens":64,"sampling":{"temperature":0.8}}`,
         *                    or `null` for load-time defaults.
         * @return The generated continuation (not the prompt).
         * @throws IllegalStateException on generation failure.
         */
        fun generate(prompt: String, optionsJson: String? = null): String {
            val out = PointerByReference()
            val status = rt.api.sdxLlmGenerate(rt.runtime, modelHandle, prompt, optionsJson, out)
            if (status != LlmStatus.OK.code) {
                throw IllegalStateException(
                    "sdxLlmGenerate failed (${LlmStatus.label(status)}): ${rt.lastError()}")
            }
            return readAndFree(out.value)
        }

        // ── Stats + info ──────────────────────────────────────────────────────

        /**
         * Returns a [LlmResultStats] parsed from the JSON stats of the most
         * recent [generate] call (token counts, tok/s, finish reason).
         */
        fun lastResultStats(): LlmResultStats {
            val json = lastResultJson()
            return LlmResultStats(
                rawJson      = json,
                promptTokens = extractInt(json, "promptTokens"),
                newTokens    = extractInt(json, "newTokens"),
                tokensPerSec = extractDouble(json, "tokensPerSec"),
                finishReason = extractString(json, "finishReason"),
            )
        }

        /** Raw JSON string from `sdxLlmLastResultJson`. Callers normally use [lastResultStats]. */
        fun lastResultJson(): String {
            val out = PointerByReference()
            val status = rt.api.sdxLlmLastResultJson(rt.runtime, modelHandle, out)
            if (status != LlmStatus.OK.code) {
                throw IllegalStateException(
                    "sdxLlmLastResultJson failed (${LlmStatus.label(status)}): ${rt.lastError()}")
            }
            return readAndFree(out.value)
        }

        /** JSON summary of this model: inputs, outputs, vocab size, chat template flag. */
        fun infoJson(): String {
            val out = PointerByReference()
            val status = rt.api.sdxLlmInfoJson(rt.runtime, modelHandle, out)
            if (status != LlmStatus.OK.code) {
                throw IllegalStateException(
                    "sdxLlmInfoJson failed (${LlmStatus.label(status)}): ${rt.lastError()}")
            }
            return readAndFree(out.value)
        }

        // ── Tokenization ──────────────────────────────────────────────────────

        /**
         * Encodes [text] to token IDs.
         *
         * @param text             Input text (UTF-8).
         * @param addSpecialTokens `true` to prepend/append BOS/EOS tokens.
         * @return IntArray of token IDs.
         */
        fun tokenize(text: String, addSpecialTokens: Boolean = false): IntArray {
            val outIds   = PointerByReference()
            val outCount = IntByReference()
            val status = rt.api.sdxLlmTokenize(
                rt.runtime, modelHandle, text,
                if (addSpecialTokens) 1 else 0,
                outIds, outCount,
            )
            if (status != LlmStatus.OK.code) {
                throw IllegalStateException(
                    "sdxLlmTokenize failed (${LlmStatus.label(status)}): ${rt.lastError()}")
            }
            val count  = outCount.value
            val idsPtr = outIds.value
            val ids    = idsPtr.getIntArray(0, count)
            rt.api.sdxLlmFree(rt.runtime, idsPtr)
            return ids
        }

        /**
         * Decodes [ids] to text.
         *
         * @param ids                Token ID array.
         * @param skipSpecialTokens  `true` to omit BOS/EOS/pad from output.
         * @return Decoded text (UTF-8).
         */
        fun detokenize(ids: IntArray, skipSpecialTokens: Boolean = true): String {
            val out = PointerByReference()
            val status = rt.api.sdxLlmDetokenize(
                rt.runtime, modelHandle,
                ids, ids.size,
                if (skipSpecialTokens) 1 else 0,
                out,
            )
            if (status != LlmStatus.OK.code) {
                throw IllegalStateException(
                    "sdxLlmDetokenize failed (${LlmStatus.label(status)}): ${rt.lastError()}")
            }
            return readAndFree(out.value)
        }

        // ── Resource management ───────────────────────────────────────────────

        override fun close() {
            rt.api.sdxLlmUnloadModel(rt.runtime, modelHandle)
        }

        // ── Internal helpers ──────────────────────────────────────────────────

        /** Reads the NUL-terminated string at [ptr], frees it, and returns a Kotlin String. */
        private fun readAndFree(ptr: Pointer?): String {
            if (ptr == null) return ""
            return try {
                ptr.getString(0, StandardCharsets.UTF_8.name()) ?: ""
            } finally {
                rt.api.sdxLlmFree(rt.runtime, ptr)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VLM / audio top-level helpers (stateless — no persistent model handle)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * VLM document extraction (SmolDocling): image/PDF → doctags/markdown/text.
 *
 * Stateless — loads and releases the model per call. Use for one-off
 * extractions; for batches, prefer the future stateful VLM API.
 *
 * @param runtime       An open [SdxLlmRuntime].
 * @param modelPath     Path to the SmolDocling model directory.
 * @param tokenizerPath Path to `tokenizer.json` or `null`.
 * @param inputPath     Path to an image (PNG/JPEG) or PDF file.
 * @param optionsJson   e.g. `{"maxNewTokens":512,"format":"doctags"}`.
 * @return Extracted text in the requested format.
 */
fun vlmExtract(
    runtime: SdxLlmRuntime,
    modelPath: String,
    tokenizerPath: String? = null,
    inputPath: String,
    optionsJson: String? = null,
): String {
    val out = PointerByReference()
    val status = runtime.api.sdxVlmExtract(
        runtime.runtime, modelPath, tokenizerPath, inputPath, optionsJson, out)
    if (status != LlmStatus.OK.code) {
        throw IllegalStateException(
            "sdxVlmExtract failed (${LlmStatus.label(status)}): ${runtime.lastError()}")
    }
    val ptr = out.value ?: return ""
    return try {
        ptr.getString(0, StandardCharsets.UTF_8.name()) ?: ""
    } finally {
        runtime.api.sdxLlmFree(runtime.runtime, ptr)
    }
}

/**
 * Whisper speech-to-text — stateless, loads and releases per call.
 *
 * @param runtime      An open [SdxLlmRuntime].
 * @param modelPath    Path to a Whisper ONNX model directory.
 * @param audioPath    Path to a WAV file.
 * @param optionsJson  e.g. `{"language":"en","maxNewTokens":448}`.
 * @return Transcription text (UTF-8).
 */
fun audioTranscribe(
    runtime: SdxLlmRuntime,
    modelPath: String,
    audioPath: String,
    optionsJson: String? = null,
): String {
    val out = PointerByReference()
    val status = runtime.api.sdxAudioTranscribe(
        runtime.runtime, modelPath, audioPath, optionsJson, out)
    if (status != LlmStatus.OK.code) {
        throw IllegalStateException(
            "sdxAudioTranscribe failed (${LlmStatus.label(status)}): ${runtime.lastError()}")
    }
    val ptr = out.value ?: return ""
    return try {
        ptr.getString(0, StandardCharsets.UTF_8.name()) ?: ""
    } finally {
        runtime.api.sdxLlmFree(runtime.runtime, ptr)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON extraction helpers (no external JSON library dependency)
// ─────────────────────────────────────────────────────────────────────────────

private val INT_PATTERN    = Regex(""""(\w+)"\s*:\s*(-?\d+)""")
private val DOUBLE_PATTERN = Regex(""""(\w+)"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)""")
private val STRING_PATTERN = Regex(""""(\w+)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"""")

internal fun extractInt(json: String, key: String): Int? =
    INT_PATTERN.findAll(json).firstOrNull { it.groupValues[1] == key }
        ?.groupValues?.get(2)?.toIntOrNull()

internal fun extractDouble(json: String, key: String): Double? =
    DOUBLE_PATTERN.findAll(json).firstOrNull { it.groupValues[1] == key }
        ?.groupValues?.get(2)?.toDoubleOrNull()

internal fun extractString(json: String, key: String): String? =
    STRING_PATTERN.findAll(json).firstOrNull { it.groupValues[1] == key }
        ?.groupValues?.get(2)
