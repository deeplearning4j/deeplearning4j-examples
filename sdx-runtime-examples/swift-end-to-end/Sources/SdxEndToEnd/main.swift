// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

// End-to-end walkthrough of the SDX runtime from Swift — no JVM involved.
//
// Loads models/mlp.sdz through the SDX C ABI via the SdxRuntime Swift wrapper
// and demonstrates the full SDK lifecycle:
//
//   1. Runtime and model creation
//   2. Input-contract discovery via inputNames()
//   3. Placeholder marking for the batch input
//   4. Warmup runs (phase → SHAPES_FROZEN)
//   5. freezeShapes() → DSP replay fast path (phase → REPLAYING)
//   6. Execution-report telemetry via SdxExecutionReport
//   7. Canonical output verification (≤ 1e-4)
//   8. Error-path demonstration
//
// External inputs are positional in plan order; inputNames() reveals the
// expected names.  For models/mlp.sdz the order is:
//
//   input[0] = "w1"   shape [4,8]   — first-layer weight
//   input[1] = "b1"   shape [8]     — first-layer bias
//   input[2] = "w2"   shape [8,3]   — second-layer weight
//   input[3] = "b2"   shape [3]     — second-layer bias
//   input[4] = "x"    shape [2,4]   — batch input (placeholder)
//
// Output: "probs"  shape [2,3]  — per-class softmax probabilities.
//
// Build/run:
//   swift run -Xlinker -L/path/to/sdk/lib SdxEndToEnd [path/to/mlp.sdz]

import Foundation
import SdxRuntime

// ── Constants ─────────────────────────────────────────────────────────────────

/// Canonical verification vector for models/mlp.sdz (from GenerateExampleModel).
let canonicalX: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

/// Expected softmax probabilities for canonicalX fed through models/mlp.sdz.
/// Tolerance: maxDiff ≤ 1e-4.
///
///   row 0: [0.44481823, 0.3220363,  0.23314552]
///   row 1: [0.4567148,  0.31961477, 0.22367041]
let expectedProbs: [Float] = [
    0.44481823, 0.3220363,  0.23314552,
    0.4567148,  0.31961477, 0.22367041,
]

// ── Weight factory ────────────────────────────────────────────────────────────

/// Return the weight tensor for the given plan-input name.
///
/// In production, weights travel *inside* the `.sdz` bundle.  This example
/// supplies them externally to show the full binding contract.  The values are
/// deterministic linspace initialisers that match the fixture.
func weightTensor(named name: String) -> SdxTensor? {
    func linspace(_ start: Float, _ end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        return (0..<count).map { start + (end - start) * Float($0) / Float(count - 1) }
    }

    switch name {
    case "w1": return SdxTensor(shape: [4, 8], scalars: linspace(-1.0,  1.0, count: 32))
    case "b1": return SdxTensor(shape: [8],    scalars: linspace( 0.0,  0.7, count:  8))
    case "w2": return SdxTensor(shape: [8, 3], scalars: linspace( 1.0, -1.0, count: 24))
    case "b2": return SdxTensor(shape: [3],    scalars: linspace(-0.1,  0.1, count:  3))
    default:   return nil
    }
}

// ── Per-step verification ─────────────────────────────────────────────────────

/// Build the input dictionary for one inference step, run the plan, verify
/// outputs, and print a status line.
///
/// - Parameters:
///   - ctx:        The inference context.
///   - inputNames: Plan input names in binding order (from ``SdxContext/inputNames()``).
///   - step:       Step counter (step 1 uses the canonical x vector for exact verification).
/// - Returns: `true` when all checks pass.
@discardableResult
func runStep(_ ctx: SdxContext, inputNames: [String], step: Int) throws -> Bool {
    // Scale canonical x so values differ between steps while shape stays fixed.
    // Step 1 uses the exact canonical vector so the baked expectation applies.
    let xScalars = canonicalX.map { $0 * Float(step) }
    let x = SdxTensor(shape: [2, 4], scalars: xScalars)

    // Build the input dictionary — weights are constant; x changes per step.
    var inputs: [String: SdxTensor] = [:]
    for name in inputNames {
        if name == "x" {
            inputs[name] = x
        } else {
            guard let w = weightTensor(named: name) else {
                fatalError("No weight value for plan input '\(name)'")
            }
            inputs[name] = w
        }
    }

    // Run — output shape is [2, 3] (batch=2, classes=3).
    let outputs = try ctx.run(inputs: inputs, outputShapes: ["probs": [2, 3]])
    let probs   = outputs["probs"]!.scalars

    // Each row of the softmax output must sum to 1.
    let rowSumsOk = abs(probs[0] + probs[1] + probs[2] - 1) <= 1e-5
                 && abs(probs[3] + probs[4] + probs[5] - 1) <= 1e-5

    var passed = rowSumsOk
    var details = "rows sum to 1: \(rowSumsOk)"

    // On step 1 also verify the exact canonical expectation.
    if step == 1 {
        let maxDiff = zip(probs, expectedProbs).map { abs($0 - $1) }.max() ?? 0
        let matchesCanonical = maxDiff <= 1e-4
        passed = passed && matchesCanonical
        details += "; matches canonical: \(matchesCanonical) (maxDiff=\(maxDiff))"
    }

    let phase = ctx.planPhase?.description ?? "?"
    print("  step \(step): phase=\(phase)  execCount=\(ctx.executionCount())  \(details)")
    return passed
}

// ── Main ──────────────────────────────────────────────────────────────────────

let modelPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "../models/mlp.sdz"

guard FileManager.default.fileExists(atPath: modelPath) else {
    fputs("Model not found: \(modelPath)\n" +
          "Generate it with the java-end-to-end GenerateExampleModel tool.\n",
          stderr)
    exit(2)
}

do {
    // ── Step 1: runtime + model ───────────────────────────────────────────────
    print("== Step 1: create the runtime and load \(modelPath) ==")
    let runtime = try SdxRuntime()
    print("SDX ABI version: \(runtime.abiVersion())")

    let model = try runtime.loadModel(path: modelPath)
    let ctx   = try model.createContext(requestedOutputs: ["probs"])

    // ── Step 2: input-contract discovery ─────────────────────────────────────
    print("\n== Step 2: discover the plan's input contract ==")
    let names = ctx.inputNames()
    print("Plan expects \(ctx.numInputs()) external input(s), \(ctx.numOutputs()) output(s):")
    for (i, name) in names.enumerated() {
        print("  input[\(i)] = \"\(name)\"")
    }

    // Mark the batch-data input as a placeholder so the runtime syncs it on
    // every run (value and potentially shape may change).
    if let xIndex = names.firstIndex(of: "x") {
        try ctx.markInputPlaceholder(Int32(xIndex))
        print("Marked input[\(xIndex)] \"x\" as placeholder.")
    }

    // ── Step 3: warmup runs ───────────────────────────────────────────────────
    print("\n== Step 3: warmup runs (shapes stabilise → SHAPES_FROZEN) ==")
    for step in 1...3 {
        guard try runStep(ctx, inputNames: names, step: step) else {
            fputs("FAILURE at warmup step \(step)\n", stderr); exit(1)
        }
    }

    // ── Step 4: freeze + replay ───────────────────────────────────────────────
    print("\n== Step 4: freezeShapes() → DSP replay fast path ==")
    try ctx.freezeShapes()
    print("Plan phase after freeze: \(ctx.planPhase?.description ?? "?")")
    for step in 4...6 {
        guard try runStep(ctx, inputNames: names, step: step) else {
            fputs("FAILURE at replay step \(step)\n", stderr); exit(1)
        }
    }

    // ── Step 5: execution-report telemetry ────────────────────────────────────
    print("\n== Step 5: execution report ==")
    let report = try ctx.executionReport()
    print(report)

    // ── Step 6: explicit resource release ─────────────────────────────────────
    // deinit handles this automatically; explicit close() is useful when you
    // want deterministic teardown order before a new context is created.
    ctx.close()
    model.close()

    // ── Step 7: error-path demonstration ─────────────────────────────────────
    print("== Step 7: error handling ==")
    do {
        _ = try runtime.loadModel(path: "/definitely/not/a/model.sdz")
        fputs("FAILURE: bogus load unexpectedly succeeded\n", stderr)
        exit(1)
    } catch let error as SdxError {
        print("Loading a bogus path raised: \(error)")
    }

    runtime.close()
    print("\nSUCCESS: SDX C ABI outputs verified from pure Swift (no JVM).")

} catch let error as SdxError {
    fputs("FAILURE: \(error)\n", stderr)
    exit(1)
}
