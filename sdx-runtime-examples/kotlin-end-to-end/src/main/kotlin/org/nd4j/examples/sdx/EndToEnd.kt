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

import org.nd4j.dsp.runtime.FloatTensor
import org.nd4j.dsp.runtime.KotlinSdxRuntime
import org.nd4j.dsp.runtime.PlanPhase
import org.nd4j.dsp.runtime.isReplaying
import org.nd4j.dsp.runtime.phaseLabel
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.system.exitProcess

/**
 * End-to-end walkthrough of the SDX runtime from Kotlin — no ND4J on the
 * classpath, just JNA and the SDX Kotlin facade.
 *
 * Loads the shared `models/mlp.sdz` fixture (generated once by the Java
 * `GenerateExampleModel` tool) through the SDX C ABI via [KotlinSdxRuntime],
 * and demonstrates the full SDK lifecycle:
 *
 * 1. **Create** runtime → load model → create context.
 * 2. **Discover** the plan's input contract via [KotlinSdxRuntime.KotlinSdxContext.inputNames].
 * 3. **Mark placeholders** (the `"x"` input) before the first run.
 * 4. **Warmup** runs in [PlanPhase.SLOT_BY_SLOT] mode.
 * 5. **Freeze shapes** to engage the DSP replay fast path.
 * 6. **Report** telemetry via [KotlinSdxRuntime.KotlinSdxContext.executionReport].
 * 7. **Verify** outputs against the canonical expectation (≤ 1 × 10⁻⁴).
 * 8. **Error handling** — bogus model path raises [IllegalStateException].
 *
 * The model's external inputs cover constants, variables (weights), and
 * placeholders, discovered positionally via [KotlinSdxRuntime.KotlinSdxContext.inputNames]:
 * `w1[4,8]`, `b1[8]`, `w2[8,3]`, `b2[3]`, `x[batch,4]` — all `float32`.
 * Output: `probs[batch,3]`.
 *
 * ## Run
 *
 * ```bash
 * gradle run --args="path/to/mlp.sdz"
 * ```
 */

// ─────────────────────────────────────────────────────────────────────────────
// Canonical test fixture
// ─────────────────────────────────────────────────────────────────────────────

/** Canonical input for `models/mlp.sdz`: two rows of [0.1 .. 0.8]. */
private val CANONICAL_X = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)

/**
 * Expected softmax output for [CANONICAL_X] (tolerance ≤ 1×10⁻⁴):
 * - row 0: `[0.44481823, 0.32203630, 0.23314552]`
 * - row 1: `[0.45671480, 0.31961477, 0.22367041]`
 */
private val EXPECTED_PROBS = floatArrayOf(
    0.44481823f, 0.3220363f,  0.23314552f,
    0.4567148f,  0.31961477f, 0.22367041f,
)

private const val TOLERANCE = 1e-4f

// ─────────────────────────────────────────────────────────────────────────────
// Weight provider (deterministic linspace initializers matching mlp.sdz)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns the [FloatTensor] for a named model weight.
 *
 * In a production client these values come from the model provider;
 * this example uses deterministic linspace initializers that match the
 * fixture embedded in `models/mlp.sdz`.
 */
private fun weightTensor(name: String): FloatTensor = when (name) {
    "w1" -> FloatTensor(linspace(start = -1f, end =  1f, n = 32), longArrayOf(4, 8))
    "b1" -> FloatTensor(linspace(start =  0f, end =  0.7f, n = 8), longArrayOf(8))
    "w2" -> FloatTensor(linspace(start =  1f, end = -1f, n = 24), longArrayOf(8, 3))
    "b2" -> FloatTensor(linspace(start = -0.1f, end = 0.1f, n = 3), longArrayOf(3))
    else -> error("no weight value for plan input '$name'")
}

private fun linspace(start: Float, end: Float, n: Int): FloatArray =
    FloatArray(n) { i -> start + (end - start) * i / (n - 1) }

// ─────────────────────────────────────────────────────────────────────────────
// Per-step run + verification
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Result of a single inference step.
 *
 * @param step         Step number (1-indexed).
 * @param probs        Softmax output — shape `[2, 3]`, row-major.
 * @param rowSumsValid Both rows sum to 1 within 1×10⁻⁵.
 * @param canonicalOk  `true` for step 1 only: max deviation ≤ [TOLERANCE].
 * @param maxDiff      Max absolute deviation from [EXPECTED_PROBS] (step 1 only).
 * @param phaseLabel   Human-readable DSP plan phase at execution time.
 * @param executionCount Total executions on the context at this point.
 */
data class StepResult(
    val step: Int,
    val probs: FloatArray,
    val rowSumsValid: Boolean,
    val canonicalOk: Boolean,
    val maxDiff: Float,
    val phaseLabel: String,
    val executionCount: Int,
) {
    /** `true` when all checks pass for this step. */
    val ok: Boolean get() = rowSumsValid && canonicalOk

    fun summary(): String = buildString {
        append("  run %d: phase=%-17s execCount=%3d  rows∑=1: %s"
            .format(step, phaseLabel, executionCount, if (rowSumsValid) "✓" else "✗"))
        if (step == 1) {
            append("  canonical: ${if (canonicalOk) "✓" else "✗"} (maxDiff=%.2e)".format(maxDiff))
        }
    }

    // FloatArray needs manual equals/hashCode inside a data class.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is StepResult) return false
        return step == other.step && probs.contentEquals(other.probs) &&
            rowSumsValid == other.rowSumsValid && canonicalOk == other.canonicalOk
    }

    override fun hashCode(): Int = 31 * step + probs.contentHashCode()
}

private fun KotlinSdxRuntime.KotlinSdxContext.stepRun(
    inputNames: List<String>,
    step: Int,
): StepResult {
    // Vary the input each step so values change while the shape stays fixed.
    // Step 1 uses the canonical vector exactly so the baked expectation applies.
    val xData = FloatArray(CANONICAL_X.size) { CANONICAL_X[it] * step }
    val xTensor = FloatTensor(xData, longArrayOf(2, 4))

    val weights: Map<String, FloatTensor> = inputNames
        .filterNot { it == "x" }
        .associateWith { weightTensor(it) }

    val output = FloatTensor.zeros(longArrayOf(2, 3))

    runNamed(
        inputs  = mapOf("x" to xTensor),
        weights = weights,
        outputs = listOf(output),
    )

    val probs = output.readBack()

    val rowSumsValid =
        abs(probs[0] + probs[1] + probs[2] - 1f) <= 1e-5f &&
        abs(probs[3] + probs[4] + probs[5] - 1f) <= 1e-5f

    val maxDiff = if (step == 1) {
        EXPECTED_PROBS.indices.fold(0f) { acc, i -> max(acc, abs(probs[i] - EXPECTED_PROBS[i])) }
    } else {
        0f
    }

    return StepResult(
        step           = step,
        probs          = probs,
        rowSumsValid   = rowSumsValid,
        canonicalOk    = step != 1 || maxDiff <= TOLERANCE,
        maxDiff        = maxDiff,
        phaseLabel     = phaseLabel,
        executionCount = executionCount,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fun main(args: Array<String>) {
    val modelPath = File(args.getOrElse(0) { "../models/mlp.sdz" }).absoluteFile
    require(modelPath.exists()) {
        "Model not found: $modelPath — generate it with the java-end-to-end GenerateExampleModel tool."
    }

    println("=== SDX Kotlin end-to-end ===\n")

    KotlinSdxRuntime.create().use { runtime ->
        println("Step 1 — create runtime + load ${modelPath.name}")
        println("  ABI version: ${runtime.abiVersion()}")

        runtime.loadModel(bundlePath = modelPath.path).use { model ->
            model.createContext(outputs = listOf("probs")).use { ctx ->

                // ── Discover input contract ───────────────────────────────────
                println("\nStep 2 — discover input contract")
                val inputNames = ctx.inputNames()
                println("  ${ctx.numInputs} external inputs, ${ctx.numOutputs} output(s):")
                inputNames.forEachIndexed { i, name -> println("  input[$i] = \"$name\"") }

                // ── Mark placeholders ─────────────────────────────────────────
                val marked = ctx.markPlaceholders("x")
                println("  Marked as placeholder: $marked (indices)")

                // ── Warmup ────────────────────────────────────────────────────
                println("\nStep 3 — warmup runs (${PlanPhase.SLOT_BY_SLOT.name})")
                for (step in 1..3) {
                    val result = ctx.stepRun(inputNames, step)
                    println(result.summary())
                    check(result.ok) { "Step $step failed: $result" }
                }

                // ── Freeze + replay ───────────────────────────────────────────
                println("\nStep 4 — freezeShapes() → fast-path replay")
                ctx.freezeShapes()
                println("  Phase after freeze: ${ctx.phaseLabel}")

                for (step in 4..6) {
                    val result = ctx.stepRun(inputNames, step)
                    println(result.summary())
                    check(result.ok) { "Step $step failed: $result" }
                }

                val replayStatus = if (ctx.isReplaying) "active ✓" else "not active (${ctx.phaseLabel})"
                println("  Graph replay: $replayStatus")

                // ── Execution report ──────────────────────────────────────────
                println("\nStep 5 — execution report")
                println(ctx.executionReport().summary()
                    .lines().joinToString("\n") { "  $it" }.trimEnd())
            }
        }

        // ── Error handling ────────────────────────────────────────────────────
        println("\nStep 6 — error path (bogus model)")
        runCatching {
            runtime.loadModel("/definitely/not/a/model.sdz")
        }.onSuccess {
            System.err.println("UNEXPECTED: bogus load succeeded")
            exitProcess(1)
        }.onFailure { e ->
            println("  Caught ${e::class.simpleName}: ${e.message}")
        }
    }

    println("\nSUCCESS — SDX C ABI outputs verified from Kotlin (no ND4J classpath).")
}
