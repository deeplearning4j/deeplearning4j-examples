# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""LLM/VLM/STT end-to-end walkthrough using the AOT ``libsdx_llm`` C ABI.

Demonstrates the full lifecycle of the SDX LLM surface — no JVM required:

1. Runtime / model creation via context managers.
2. Model info JSON query (``SdxLlmModel.info()``).
3. Tokenize → detokenize round-trip.
4. Text generation with greedy sampling and ``GenerateStats`` telemetry.
5. Canonical output assertion: generated text must contain ``" Paris."``.
6. Optional: VLM document extraction (``--vlm``, gated on asset existence).
7. Optional: Whisper STT transcription (``--transcribe``, gated on asset).

Running
-------
::

    # Point the loader at the unpacked SDK package (contains lib/libsdx_llm.so):
    export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8

    # Model paths default to the local cache locations used during development;
    # override them with env vars:
    #   SDX_LLM_MODEL_PATH   — .gguf model file
    #   SDX_LLM_TOKENIZER    — tokenizer.json file (or directory)
    #   SDX_VLM_MODEL_PATH   — SmolDocling model directory (for --vlm)
    #   SDX_VLM_IMAGE_PATH   — image/PDF for VLM (for --vlm)
    #   SDX_STT_MODEL_PATH   — Whisper model directory (for --transcribe)
    #   SDX_STT_AUDIO_PATH   — .wav file (for --transcribe)

    /usr/bin/python3 llm_example.py [--vlm] [--transcribe]

Use the system Python (not conda) to avoid libstdc++ version conflicts.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Step output should stream even when piped (model loads block for minutes;
# block-buffered stdout would otherwise show nothing until exit).
sys.stdout.reconfigure(line_buffering=True)

# ── Wrapper import ───────────────────────────────────────────────────────────

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sdx_llm import (  # noqa: E402
    GenerateStats,
    LlmStatus,
    SdxLlmError,
    SdxLlmRuntime,
    load_library,
)

# ── Default asset paths ──────────────────────────────────────────────────────
# All paths default to the local development cache; override via env vars.

_HOME = pathlib.Path.home()

DEFAULT_MODEL = str(
    _HOME / ".cache" / "dl4j-llm-models" / "Qwen3.5-0.8B-Q4_K_M.gguf"
)
DEFAULT_TOKENIZER = str(
    _HOME / ".cache" / "dl4j-llm-models" / "qwen35-0.8B-tokenizer.json"
)
DEFAULT_VLM_MODEL = str(
    _HOME / ".kompile" / "models" / "vlm" / "smoldocling-256m"
)
DEFAULT_VLM_IMAGE = "/tmp/vlm-agent-test-doc.png"
DEFAULT_STT_MODEL = str(
    _HOME / ".cache" / "dl4j-whisper-models" / "whisper-tiny-onnx"
)
DEFAULT_STT_AUDIO = "/tmp/test-audio-agent.wav"

MODEL_PATH     = os.environ.get("SDX_LLM_MODEL_PATH",   DEFAULT_MODEL)
TOKENIZER_PATH = os.environ.get("SDX_LLM_TOKENIZER",    DEFAULT_TOKENIZER)
VLM_MODEL_PATH = os.environ.get("SDX_VLM_MODEL_PATH",   DEFAULT_VLM_MODEL)
VLM_IMAGE_PATH = os.environ.get("SDX_VLM_IMAGE_PATH",   DEFAULT_VLM_IMAGE)
STT_MODEL_PATH = os.environ.get("SDX_STT_MODEL_PATH",   DEFAULT_STT_MODEL)
STT_AUDIO_PATH = os.environ.get("SDX_STT_AUDIO_PATH",   DEFAULT_STT_AUDIO)

# The canonical generate prompt and the substring that must appear in the output.
CANON_PROMPT   = "The capital of France is"
CANON_EXPECTED = " Paris."

# Generation options: 8 new tokens, greedy sampling.
GENERATE_OPTIONS = '{"maxNewTokens":8,"sampling":{"preset":"greedy"}}'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_asset(path: str, label: str) -> bool:
    """Return True if the path exists; print a skip message and return False otherwise."""
    if pathlib.Path(path).exists():
        return True
    print(f"  [SKIP] {label} not found at: {path}")
    return False


def _pass_fail(ok: bool, label: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_info(model) -> None:
    """Step 2: query model/tokenizer summary."""
    print("\n== Step 2: model info ==")
    info = model.info()
    if info:
        print(f"  vocab_size         = {info.get('vocabSize', info.get('vocab_size', '?'))}")
        print(f"  has_chat_template  = {info.get('hasChatTemplate', info.get('has_chat_template', '?'))}")
    else:
        print("  (no info returned)")


def step_tokenize(model) -> bool:
    """Step 3: tokenize → detokenize round-trip; return True on success."""
    print("\n== Step 3: tokenize / detokenize round-trip ==")
    text = "Hello, world!"
    ids = model.tokenize(text, add_special_tokens=False)
    print(f"  '{text}' → token IDs: {ids}")
    recovered = model.detokenize(ids, skip_special_tokens=True)
    print(f"  IDs → '{recovered}'")
    ok = text.strip().lower() in recovered.strip().lower() or recovered.strip() in text
    _pass_fail(ok, "round-trip matches source text")
    return ok


def step_generate(model) -> bool:
    """Step 4+5: generate text and verify canonical output; return True on success."""
    print(f"\n== Step 4: generate (prompt={CANON_PROMPT!r}, options={GENERATE_OPTIONS}) ==")
    print("  (first call includes GGUF import + pipeline warmup; may take 1-3 min)")
    text = model.generate(CANON_PROMPT, options_json=GENERATE_OPTIONS)
    print(f"  generated text: {text!r}")

    stats: GenerateStats = model.last_result()
    print("\n== Step 5: GenerateStats (from sdxLlmLastResultJson) ==")
    print(f"  prompt_tokens    = {stats.prompt_tokens}")
    print(f"  generated_tokens = {stats.generated_tokens}")
    print(f"  generation_time_ms      = {stats.generation_time_ms:.1f}")
    print(f"  first_token_latency_ms  = {stats.first_token_latency_ms:.1f}")
    print(f"  tokens_per_sec          = {stats.tokens_per_sec:.2f}")
    print(f"  finish_reason    = {stats.finish_reason!r}")

    ok = CANON_EXPECTED in text
    _pass_fail(ok, f"generated text contains {CANON_EXPECTED!r}")
    return ok


def step_vlm(runtime, *, require: bool = False) -> bool:
    """Step 6 (optional): VLM document extraction."""
    print("\n== Step 6 (optional): VLM document extraction ==")
    assets_ok = (
        _check_asset(VLM_MODEL_PATH, "VLM model") and
        _check_asset(VLM_IMAGE_PATH, "VLM test image")
    )
    if not assets_ok:
        if require:
            print("  FAIL: --vlm requested but assets missing.")
            return False
        print("  [SKIP] VLM assets not available — skipping.")
        return True

    print(f"  model : {VLM_MODEL_PATH}")
    print(f"  image : {VLM_IMAGE_PATH}")
    print("  (stateless — loads model per call; may take 1-3 min)")
    text = runtime.vlm_extract(
        VLM_MODEL_PATH,
        None,
        VLM_IMAGE_PATH,
        options_json='{"maxNewTokens":256,"format":"doctags"}',
    )
    print(f"  extraction result (first 200 chars): {text[:200]!r}")
    ok = bool(text.strip())
    _pass_fail(ok, "VLM returned non-empty text")
    return ok


def step_transcribe(runtime, *, require: bool = False) -> bool:
    """Step 7 (optional): Whisper speech-to-text."""
    print("\n== Step 7 (optional): Whisper transcription ==")
    assets_ok = (
        _check_asset(STT_MODEL_PATH, "Whisper model") and
        _check_asset(STT_AUDIO_PATH, "test audio")
    )
    if not assets_ok:
        if require:
            print("  FAIL: --transcribe requested but assets missing.")
            return False
        print("  [SKIP] STT assets not available — skipping.")
        return True

    print(f"  model : {STT_MODEL_PATH}")
    print(f"  audio : {STT_AUDIO_PATH}")
    print("  (stateless — loads model per call)")
    text = runtime.audio_transcribe(
        STT_MODEL_PATH,
        STT_AUDIO_PATH,
        options_json='{"language":"en"}',
    )
    print(f"  transcript: {text!r}")
    ok = bool(text.strip())
    _pass_fail(ok, "Whisper returned non-empty transcript")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SDX LLM end-to-end example (AOT libsdx_llm, no JVM)"
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Run VLM document extraction (requires SmolDocling model + test image)",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Run Whisper STT transcription (requires Whisper model + test audio)",
    )
    args = parser.parse_args()

    # ── Pre-flight: verify core model assets ──────────────────────────────────
    print("== Step 0: pre-flight checks ==")
    if not _check_asset(MODEL_PATH, "LLM model (.gguf)"):
        print(
            f"\nSet SDX_LLM_MODEL_PATH to an existing .gguf model file.\n"
            f"Current default: {MODEL_PATH}"
        )
        return 2
    if not _check_asset(TOKENIZER_PATH, "tokenizer.json"):
        print(
            f"\nSet SDX_LLM_TOKENIZER to an existing tokenizer.json file.\n"
            f"Current default: {TOKENIZER_PATH}"
        )
        return 2

    # Load the shared library (raises if SDX_LLM_AOT_HOME is not set).
    try:
        load_library()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nLibrary load failed: {exc}")
        return 2

    all_ok = True

    # ── Step 1: create runtime and load model ─────────────────────────────────
    print("\n== Step 1: create runtime and load model ==")
    print(f"  model      : {MODEL_PATH}")
    print(f"  tokenizer  : {TOKENIZER_PATH}")

    try:
        with SdxLlmRuntime() as runtime:
            print(f"  ABI version: {runtime.abi_version()}")

            # Load-time options: greedy sampling by default; can be overridden
            # per generate() call.
            load_opts = '{"graphOptimizer":true,"sampling":{"preset":"greedy"}}'
            with runtime.load_model(MODEL_PATH, TOKENIZER_PATH, load_opts) as model:

                step_info(model)
                all_ok &= step_tokenize(model)
                all_ok &= step_generate(model)

            # ── Optional steps (stateless — no loaded model needed). Strictly
            # flag-gated: each loads its own model, adding many minutes on CPU,
            # so the default run stays a quick LLM smoke. ──────────────────────
            if args.vlm:
                all_ok &= step_vlm(runtime, require=True)
            else:
                print("\n== Step 6 (optional): VLM document extraction — skipped (pass --vlm) ==")

            if args.transcribe:
                all_ok &= step_transcribe(runtime, require=True)
            else:
                print("\n== Step 7 (optional): Whisper transcription — skipped (pass --transcribe) ==")

            # ── Error-path demonstration ────────────────────────────────────────
            print("\n== Step 8: error handling ==")
            try:
                runtime.load_model("/definitely/not/a/model.gguf")
                print("  unexpected: bogus load succeeded")
                all_ok = False
            except SdxLlmError as exc:
                print(f"  Loading a bogus path raised SdxLlmError: {exc}")

    except SdxLlmError as exc:
        print(f"\nFATAL: {exc}")
        return 1

    if all_ok:
        print(
            "\nSUCCESS: SDX LLM C ABI outputs verified from pure Python (no JVM)."
        )
        return 0
    else:
        print("\nFAILURE: one or more checks did not pass.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
