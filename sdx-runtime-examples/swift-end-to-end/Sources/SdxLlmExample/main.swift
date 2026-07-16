// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

// SdxLlmExample — end-to-end LLM walkthrough using the AOT libsdx_llm C ABI.
//
// Demonstrates the full lifecycle using the SdxLlm Swift wrapper:
//
//   1. Runtime creation + ABI version check
//   2. Model load (GGUF + HuggingFace tokenizer)
//   3. Model info JSON
//   4. Tokenize / detokenize round-trip
//   5. Text generation + assertion on expected output
//   6. Generation statistics (tok/s, token counts)
//   7. Error-path demonstration
//   8. VLM and audio (optional — controlled by env vars)
//
// Build & run (AOT SDK at /tmp/sdx-cpu-v8):
//
//   export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
//   swift run -Xcc -I$SDX_LLM_AOT_HOME/include \
//             -Xlinker -L$SDX_LLM_AOT_HOME/lib \
//             SdxLlmExample \
//             ~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//             ~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json
//
// THREADING NOTE: the GraalVM isolate in libsdx_llm has single-thread affinity.
// Create, use, and destroy each SdxLlmRuntime from one OS thread.  For concurrent
// generation, create one runtime per thread — models are not shared.

import Foundation
import SdxLlm

// ── Helpers ───────────────────────────────────────────────────────────────────

func env(_ key: String) -> String? { ProcessInfo.processInfo.environment[key] }

func banner(_ title: String) { print("\n== \(title) ==") }

// ── Argument parsing ──────────────────────────────────────────────────────────

let args = CommandLine.arguments

// Allow arguments as positional or environment variables.
let modelPath      = args.count > 1 ? args[1] : env("SDX_LLM_MODEL")
let tokenizerPath  = args.count > 2 ? args[2] : env("SDX_LLM_TOKENIZER")
// Optional: VLM model dir (SmolDocling) and a test image
let vlmModelPath   = args.count > 3 ? args[3] : env("SDX_VLM_MODEL")
let vlmImagePath   = args.count > 4 ? args[4] : env("SDX_VLM_IMAGE")
// Optional: Whisper model dir and audio file
let whisperModel   = args.count > 5 ? args[5] : env("SDX_WHISPER_MODEL")
let audioPath      = args.count > 6 ? args[6] : env("SDX_AUDIO_PATH")

guard let modelPath else {
    fputs("""
    Usage: SdxLlmExample <model.gguf> [tokenizer.json] [vlm-model] [image] [whisper-model] [audio]
    Or set env: SDX_LLM_MODEL, SDX_LLM_TOKENIZER, SDX_LLM_AOT_HOME
    \n""", stderr)
    exit(2)
}
guard FileManager.default.fileExists(atPath: modelPath) else {
    fputs("Model not found: \(modelPath)\n", stderr)
    exit(2)
}

// ── Main walkthrough ──────────────────────────────────────────────────────────

do {
    // ── Step 1: runtime ────────────────────────────────────────────────────────
    banner("Step 1: create the LLM runtime")
    // SdxLlmRuntime() auto-sets SDX_NATIVE_LIB_DIR from SDX_LLM_AOT_HOME if unset.
    let rt = try SdxLlmRuntime()
    print("ABI version    : \(rt.abiVersion())")
    print("SDX_LLM_ABI_VERSION constant: \(SDX_LLM_ABI_VERSION)")
    guard rt.abiVersion() == SDX_LLM_ABI_VERSION else {
        fputs("FAIL: ABI version mismatch — header/library out of sync\n", stderr)
        exit(1)
    }

    // ── Step 2: load model ─────────────────────────────────────────────────────
    banner("Step 2: load model (\(URL(fileURLWithPath: modelPath).lastPathComponent))")
    let loadOpts = #"{"maxNewTokens":8,"sampling":{"preset":"greedy"}}"#
    print("Loading with options: \(loadOpts)")
    let model = try rt.loadModel(
        modelPath:     modelPath,
        tokenizerPath: tokenizerPath,
        optionsJson:   loadOpts)
    print("Model loaded successfully.")

    // ── Step 3: model info ─────────────────────────────────────────────────────
    banner("Step 3: model info JSON")
    let info = try model.infoJson()
    // Print truncated so output stays readable.
    let infoTrunc = info.count > 300 ? String(info.prefix(300)) + "…" : info
    print("info: \(infoTrunc)")

    // ── Step 4: tokenize / detokenize round-trip ───────────────────────────────
    banner("Step 4: tokenize / detokenize")
    let probe = "The capital of France is"
    let ids   = try model.tokenize(probe, addSpecialTokens: false)
    print("tokenize(\"\(probe)\") → \(ids.count) tokens: \(ids)")
    let roundtrip = try model.detokenize(ids, skipSpecialTokens: true)
    print("detokenize → \"\(roundtrip)\"")
    let rtMatch = roundtrip.trimmingCharacters(in: .whitespaces)
        .lowercased()
        .contains(probe.lowercased().trimmingCharacters(in: .whitespaces))
    print("Round-trip contains original: \(rtMatch)")
    guard rtMatch else {
        fputs("FAIL: tokenize/detokenize round-trip diverged\n", stderr)
        exit(1)
    }

    // ── Step 5: text generation ────────────────────────────────────────────────
    banner("Step 5: text generation")
    let prompt    = "The capital of France is"
    let genOpts   = #"{"maxNewTokens":8,"sampling":{"preset":"greedy"}}"#
    print("Prompt : \"\(prompt)\"")
    print("Options: \(genOpts)")
    print("Generating… (CPU import + generation takes ~1–3 min on first run)")
    let generated = try model.generate(prompt, optionsJson: genOpts)
    print("Output : \"\(generated)\"")

    // Assertion: greedy Qwen3.5-0.8B should output " Paris" within 8 tokens.
    let normGen = generated.trimmingCharacters(in: .whitespaces).lowercased()
    guard normGen.contains("paris") else {
        fputs("FAIL: expected generated text to contain \"Paris\", got: \"\(generated)\"\n", stderr)
        exit(1)
    }
    print("PASS: generated text contains \"Paris\".")

    // ── Step 6: generation statistics ─────────────────────────────────────────
    banner("Step 6: generation statistics")
    if let stats = try model.lastResultStats() {
        print(stats)
    } else {
        print("(no stats available)")
    }

    // ── Step 7: error path ─────────────────────────────────────────────────────
    banner("Step 7: error-path demonstration")
    do {
        _ = try rt.loadModel(modelPath: "/definitely/not/a/model.gguf")
        fputs("FAIL: bogus model load should have thrown\n", stderr)
        exit(1)
    } catch let e as SdxLlmError {
        print("Loading a bogus path raised: \(e)")
    }
    print("Error path: PASS.")

    // ── Step 8: optional VLM ──────────────────────────────────────────────────
    if let vmPath = vlmModelPath, let imgPath = vlmImagePath,
       FileManager.default.fileExists(atPath: vmPath),
       FileManager.default.fileExists(atPath: imgPath) {
        banner("Step 8a: VLM extraction (SmolDocling)")
        print("Image : \(imgPath)")
        let vtOpts = #"{"format":"markdown","maxNewTokens":256}"#
        let vtText = try rt.vlmExtract(
            modelPath:   vmPath,
            inputPath:   imgPath,
            optionsJson: vtOpts)
        let truncVt = vtText.count > 200 ? String(vtText.prefix(200)) + "…" : vtText
        print("VLM output: \(truncVt)")
    } else {
        print("\n(Skipping VLM — set SDX_VLM_MODEL and SDX_VLM_IMAGE to enable)")
    }

    if let wsModel = whisperModel, let audPath = audioPath,
       FileManager.default.fileExists(atPath: wsModel),
       FileManager.default.fileExists(atPath: audPath) {
        banner("Step 8b: audio transcription (Whisper)")
        print("Audio : \(audPath)")
        let wsOpts  = #"{"language":"en"}"#
        let transcript = try rt.audioTranscribe(
            modelPath:   wsModel,
            audioPath:   audPath,
            optionsJson: wsOpts)
        print("Transcript: \(transcript)")
    } else {
        print("(Skipping STT — set SDX_WHISPER_MODEL and SDX_AUDIO_PATH to enable)")
    }

    // ── Teardown ───────────────────────────────────────────────────────────────
    model.close()
    rt.close()

    print("\nSUCCESS: SDX LLM AOT C ABI verified from pure Swift (no JVM).")

} catch let e as SdxLlmError {
    fputs("FAILURE: \(e)\n", stderr)
    exit(1)
}
