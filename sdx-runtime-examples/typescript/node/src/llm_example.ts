/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * SDX LLM AOT runtime — end-to-end walkthrough from TypeScript / Node.js.
 *
 * Loads a GGUF model through the `sdx_llm_c.h` C ABI via the SdxLlmRuntime
 * wrapper (koffi FFI on a dedicated Worker thread — no JVM) and demonstrates
 * the full lifecycle:
 *
 *  1. Runtime creation + ABI version check.
 *  2. Model load (GGUF + HuggingFace tokenizer.json).
 *  3. Model info JSON.
 *  4. Tokenize / detokenize round-trip.
 *  5. Text generation + assertion on expected output.
 *  6. Generation statistics (tok/s, token counts, finish reason).
 *  7. Error-path demonstration.
 *  8. Optional: VLM extraction + audio transcription.
 *
 * The `start:llm` npm script runs `node dist/llm_example.js`.  All koffi FFI
 * calls happen on a dedicated Worker thread with 128 MB stack — the GraalVM
 * isolate inside libsdx_llm needs more stack than Node's 8 MB main-thread
 * default, and the Worker provides that.
 *
 * Run (the `start:llm` script handles this automatically):
 * ```bash
 * SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8 \
 * npm run start:llm -- \
 *   ~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
 *   ~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json
 *
 * # With optional VLM + audio:
 * SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8 \
 * npm run start:llm -- \
 *   ~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf \
 *   ~/.cache/dl4j-llm-models/qwen35-0.8B-tokenizer.json \
 *   /path/to/vlm-model  /path/to/image.png \
 *   /path/to/whisper-model  /path/to/audio.wav
 * ```
 */

import * as fs   from 'fs';
import * as path from 'path';

import {
  SdxLlmRuntime,
  SdxLlmModel,
  GenerationStats,
} from './sdx_llm';

// ── ABI version constant (aligned with sdx_llm_c.h SDX_LLM_ABI_VERSION) ──────
const EXPECTED_ABI = 1;

// ── Helpers ───────────────────────────────────────────────────────────────────

function banner(s: string): void { console.log(`\n== ${s} ==`); }

function printStats(stats: GenerationStats | null): void {
  if (!stats) { console.log('  (no stats available)'); return; }
  const tok    = stats.generatedTokens   ?? '?';
  const prompt = stats.promptTokens      ?? '?';
  const tps    = stats.tokensPerSecond != null
    ? stats.tokensPerSecond.toFixed(1) : '?';
  const finish = stats.finishReason ?? '?';
  const ms     = stats.generationTimeMs != null
    ? stats.generationTimeMs.toFixed(0) : '?';
  console.log(`  prompt tokens    = ${prompt}`);
  console.log(`  new tokens       = ${tok}`);
  console.log(`  generation time  = ${ms} ms`);
  console.log(`  tok/s            = ${tps}`);
  console.log(`  finish reason    = ${finish}`);
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<number> {
  const argv = process.argv.slice(2);

  const modelPath     = argv[0] ?? process.env.SDX_LLM_MODEL;
  const tokenizerPath = argv[1] ?? process.env.SDX_LLM_TOKENIZER ?? null;
  const vlmModelPath  = argv[2] ?? process.env.SDX_VLM_MODEL     ?? null;
  const vlmImagePath  = argv[3] ?? process.env.SDX_VLM_IMAGE     ?? null;
  const whisperModel  = argv[4] ?? process.env.SDX_WHISPER_MODEL  ?? null;
  const audioPath     = argv[5] ?? process.env.SDX_AUDIO_PATH     ?? null;

  if (!modelPath) {
    console.error(
      'Usage: npm run start:llm -- <model.gguf> [tokenizer.json]\n' +
      '       [vlm-model] [image.png] [whisper-model] [audio.wav]\n' +
      'Or set: SDX_LLM_AOT_HOME, SDX_LLM_MODEL, SDX_LLM_TOKENIZER'
    );
    return 2;
  }
  if (!fs.existsSync(modelPath)) {
    console.error(`Model not found: ${modelPath}`);
    return 2;
  }

  // ── Step 1: runtime ──────────────────────────────────────────────────────────
  banner('Step 1: create the LLM runtime');
  // SdxLlmRuntime.create() spawns a Worker with 128 MB stack and calls
  // sdxLlmCreateRuntime inside it (the GraalVM isolate needs the larger stack).
  const rt = await SdxLlmRuntime.create();
  console.log(`Library     : ${rt.libraryPath}`);
  console.log(`ABI version : ${rt.abiVersion}`);
  if (rt.abiVersion !== EXPECTED_ABI) {
    console.error(`FAIL: ABI version ${rt.abiVersion} ≠ expected ${EXPECTED_ABI}`);
    await rt.dispose();
    return 1;
  }

  let model: SdxLlmModel | null = null;
  let exitCode = 1;

  try {
    // ── Step 2: load model ─────────────────────────────────────────────────────
    banner(`Step 2: load model (${path.basename(modelPath)})`);
    const loadOpts = JSON.stringify({
      maxNewTokens: 8,
      sampling: { preset: 'greedy' },
    });
    console.log(`tokenizer   : ${tokenizerPath ?? '(auto-detect)'}`);
    console.log(`options     : ${loadOpts}`);
    console.log('Loading… (CPU import + warmup takes ~1–3 min on first run)');
    model = await rt.loadModel(modelPath, tokenizerPath, loadOpts);
    console.log('Model loaded successfully.');

    // ── Step 3: model info JSON ────────────────────────────────────────────────
    banner('Step 3: model info JSON');
    const info = await model.infoJson();
    const infoTrunc = info.length > 300 ? info.slice(0, 300) + '…' : info;
    console.log(`info: ${infoTrunc}`);

    // ── Step 4: tokenize / detokenize ──────────────────────────────────────────
    banner('Step 4: tokenize / detokenize');
    const probe = 'The capital of France is';
    const ids   = await model.tokenize(probe, /*addSpecialTokens*/ false);
    console.log(`tokenize("${probe}") → ${ids.length} tokens: [${Array.from(ids).join(',')}]`);
    const roundtrip = await model.detokenize(ids, /*skipSpecialTokens*/ true);
    console.log(`detokenize → "${roundtrip}"`);
    const rtMatch = roundtrip.trim().toLowerCase()
      .includes(probe.trim().toLowerCase());
    console.log(`Round-trip contains original: ${rtMatch}`);
    if (!rtMatch) {
      console.error('FAIL: tokenize/detokenize round-trip diverged');
      return 1;
    }

    // ── Step 5: text generation ────────────────────────────────────────────────
    banner('Step 5: text generation');
    const prompt  = 'The capital of France is';
    const genOpts = JSON.stringify({ maxNewTokens: 8, sampling: { preset: 'greedy' } });
    console.log(`prompt  : "${prompt}"`);
    console.log(`options : ${genOpts}`);
    const generated = await model.generate(prompt, genOpts);
    console.log(`output  : "${generated}"`);

    const normGen = generated.trim().toLowerCase();
    if (!normGen.includes('paris')) {
      console.error(`FAIL: expected output to contain "Paris", got: "${generated}"`);
      return 1;
    }
    console.log('PASS: generated text contains "Paris".');

    // ── Step 6: generation stats ───────────────────────────────────────────────
    banner('Step 6: generation statistics');
    printStats(await model.lastResultStats());

    exitCode = 0;

  } finally {
    if (model) await model.dispose();
  }

  // ── Step 7: error path ────────────────────────────────────────────────────────
  banner('Step 7: error-path demonstration');
  try {
    await rt.loadModel('/definitely/not/a/model.gguf');
    console.error('FAIL: bogus model load should have thrown');
    exitCode = 1;
  } catch (err) {
    console.log(`Loading a bogus path throws: ${(err as Error).message.slice(0, 200)}`);
    console.log('Error path: PASS.');
  }

  // ── Step 8: optional VLM ──────────────────────────────────────────────────────
  if (vlmModelPath && vlmImagePath &&
      fs.existsSync(vlmModelPath) && fs.existsSync(vlmImagePath)) {
    banner('Step 8a: VLM extraction (SmolDocling)');
    console.log(`image : ${vlmImagePath}`);
    try {
      const vtText = await rt.vlmExtract(
        vlmModelPath, vlmImagePath, null,
        JSON.stringify({ format: 'markdown', maxNewTokens: 256 }));
      const trunc = vtText.length > 200 ? vtText.slice(0, 200) + '…' : vtText;
      console.log(`VLM output: ${trunc}`);
    } catch (err) {
      console.error(`VLM error: ${(err as Error).message}`);
    }
  } else {
    console.log('\n(Skipping VLM — pass vlm-model and image args or set SDX_VLM_MODEL/SDX_VLM_IMAGE)');
  }

  if (whisperModel && audioPath &&
      fs.existsSync(whisperModel) && fs.existsSync(audioPath)) {
    banner('Step 8b: audio transcription (Whisper)');
    console.log(`audio : ${audioPath}`);
    try {
      const transcript = await rt.audioTranscribe(
        whisperModel, audioPath,
        JSON.stringify({ language: 'en' }));
      console.log(`Transcript: ${transcript}`);
    } catch (err) {
      console.error(`STT error: ${(err as Error).message}`);
    }
  } else {
    console.log('(Skipping STT — pass whisper-model and audio args or set SDX_WHISPER_MODEL/SDX_AUDIO_PATH)');
  }

  await rt.dispose();

  if (exitCode === 0) {
    console.log('\nSUCCESS: SDX LLM AOT C ABI verified from TypeScript/Node.js (no JVM).');
  } else {
    console.error('\nFAILURE: output verification failed.');
  }
  return exitCode;
}

main().then(process.exit).catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
