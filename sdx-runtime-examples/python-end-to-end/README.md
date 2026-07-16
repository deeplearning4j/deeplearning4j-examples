# SDX Runtime — Python end-to-end example

Loads `../models/mlp.sdz` through the SDX C ABI via the `sdx_runtime` ctypes
wrapper — no JVM in the process — and walks the full SDK lifecycle using a
numpy-first, onnxruntime-style Python API:

| Step | What it demonstrates |
|------|----------------------|
| 1 | `SdxRuntime` / `SdxModel` / `SdxContext` creation via context managers |
| 2 | Input-contract discovery via `ctx.get_inputs()` → `InputMetadata` list |
| 3 | Placeholder marking, warmup runs (SLOT_BY_SLOT → SHAPES_FROZEN) |
| 4 | `ctx.freeze_shapes()` and the DSP REPLAYING fast path |
| 5 | Dict-by-name inference via `ctx.run_named(feed_dict, [output])` |
| 6 | Structured telemetry via `ExecutionSummary` dataclass |
| 7 | Canonical output verification (≤ 1e-4) and error-path demonstration |

## Key API patterns

### onnxruntime-style input discovery

```python
meta = ctx.get_inputs()          # list[InputMetadata(name, index)]
# [InputMetadata(name='w1', index=0), ..., InputMetadata(name='x', index=4)]
```

### Dict-by-name inference (no positional bookkeeping)

```python
feed = {"x": x_arr, "w1": w1_arr, "b1": b1_arr, "w2": w2_arr, "b2": b2_arr}
probs = np.zeros((2, 3), dtype=np.float32)

report = ctx.run_named(feed, [probs])   # reorders feed to plan binding order
# probs is now filled; report is an ExecutionSummary dataclass
```

### Structured execution telemetry

```python
print(report.plan_phase_name)      # "REPLAYING"
print(report.applied_backend_name) # "AUTO"
print(report.execution_time_ms)    # 0.115
print(report.used_fallback)        # False
```

## Run

```bash
# Point the loader at a directory containing the runtime library.
# An unpacked SDK's lib/ ships the JVM-free libsdx_cpu:
SDX_RUNTIME_LIBRARY_DIR=/path/to/sdk/lib /usr/bin/python3 end_to_end.py [model.sdz]
```

The wrapper is resolved from `SDX_RUNTIME_HOME/wrappers/python` when set,
otherwise from a sibling `deeplearning4j` checkout at
`deeplearning4j/libnd4j/include/dsp/runtime/bindings/python/`.  Requires numpy.

**Use the system Python** (not a conda environment) to avoid `libstdc++`
version conflicts.  If you must use conda, install a matching libstdc++ first:

```bash
conda install -c conda-forge libstdcxx-ng
```

---

## LLM/VLM/STT example — `sdx_llm.py` + `llm_example.py`

Demonstrates the SDX LLM surface (AOT `libsdx_llm`, no JVM) through the
`sdx_llm` ctypes wrapper:

| Step | What it demonstrates |
|------|----------------------|
| 1 | `SdxLlmRuntime` / `SdxLlmModel` creation via context managers |
| 2 | Model info JSON via `model.info()` |
| 3 | `model.tokenize()` / `model.detokenize()` round-trip |
| 4 | `model.generate()` with greedy sampling |
| 5 | `GenerateStats` frozen dataclass from `model.last_result()` |
| 6 | Canonical output assertion: generated text contains `" Paris."` |
| 7 | Optional VLM document extraction (`--vlm`) |
| 8 | Optional Whisper STT transcription (`--transcribe`) |

### Quick start

```bash
# 1. Unpack the AOT SDK and point to it:
export SDX_LLM_AOT_HOME=/path/to/sdx-aot-package

# 2. Override model paths if needed (defaults use ~/.cache/dl4j-llm-models/):
# export SDX_LLM_MODEL_PATH=/path/to/model.gguf
# export SDX_LLM_TOKENIZER=/path/to/tokenizer.json

/usr/bin/python3 llm_example.py [--vlm] [--transcribe]
```

Use the **system Python** (`/usr/bin/python3`) to avoid `libstdc++` conflicts.

### LLM wrapper API reference

| Name | Type | Description |
|------|------|-------------|
| `SdxLlmRuntime` | class | Runtime lifecycle; `load_model()`, context manager; `abi_version()` |
| `SdxLlmModel`   | class | Model handle; `generate()`, `tokenize()`, `detokenize()`, `info()`, `last_result()` |
| `GenerateStats` | frozen dataclass | Post-generate telemetry (`prompt_tokens`, `tokens_per_sec`, `finish_reason`, …) |
| `LlmStatus`     | constants | Status code values and names (`OK`, `MODEL_LOAD_FAILED`, …) |
| `SdxLlmError`   | exception | Raised on non-OK status; carries `.op`, `.status`, `.detail` |
| `vlm_extract`   | function | Module-level VLM extraction (SmolDocling) |
| `audio_transcribe` | function | Module-level Whisper STT |

#### `SdxLlmRuntime`

```python
with SdxLlmRuntime() as rt:
    print(rt.abi_version())           # integer ABI version
    model = rt.load_model(
        model_path,
        tokenizer_path,               # optional; None = try model dir
        '{"maxNewTokens":128}',       # optional per-load options JSON
    )
    # or use as context manager:
    with rt.load_model(model_path, tokenizer_path) as model:
        text = model.generate("Hello")
```

#### `SdxLlmModel`

```python
text  = model.generate(prompt, options_json='{"maxNewTokens":64}')
stats = model.last_result()          # GenerateStats dataclass
info  = model.info()                 # dict from sdxLlmInfoJson
ids   = model.tokenize("hello")      # List[int]
text  = model.detokenize(ids)        # str
```

#### `GenerateStats` (frozen dataclass)

```python
print(stats.prompt_tokens)    # int
print(stats.generated_tokens) # int
print(stats.tokens_per_sec)   # float
print(stats.finish_reason)    # str  — e.g. "max_length", "eos"
```

#### Threading

A `SdxLlmRuntime` is bound to the OS thread that created it.  For concurrent
generation create one `SdxLlmRuntime` per thread.  Do **not** share a runtime
across threads.

---

## SDK wrapper API reference

The `sdx_runtime` module exposes a layered API:

| Name | Type | Description |
|------|------|-------------|
| `SdxRuntime` | class | Runtime lifecycle; `load_model()`, context manager |
| `SdxModel` | class | Model handle; `create_context()`, context manager |
| `SdxContext` | class | Inference context; `run_named()`, `get_inputs()`, `freeze_shapes()` |
| `InputMetadata` | frozen dataclass | Per-input name + positional index |
| `ExecutionSummary` | frozen dataclass | Post-run telemetry (phase, backend, timing) |
| `ModelOptions` | ctypes struct | Backend / GPU target selection at load time |
| `RunOptions` | ctypes struct | Per-run backend / GPU target override |
| `TensorView` | ctypes struct | Low-level tensor descriptor (advanced use) |
| `ExecutionReport` | ctypes struct | Raw C-struct report (use `ExecutionSummary` instead) |

### `ctx.get_inputs()` → `list[InputMetadata]`

Returns plan input metadata in binding order — mirrors
`onnxruntime.InferenceSession.get_inputs()`:

```python
for m in ctx.get_inputs():
    print(m.index, m.name)   # 0 'w1', 1 'b1', ...
```

### `ctx.run_named(input_feed, outputs, options=None)` → `ExecutionSummary`

Runs inference with inputs supplied as a name→array dict.  The method
reorders the dict to match the plan's positional binding contract so callers
never need to track index numbers.  Returns a frozen `ExecutionSummary`.

```python
report = ctx.run_named(
    {"x": x_arr, "w1": w1, "b1": b1, "w2": w2, "b2": b2},
    [probs_buffer],
)
```

Raises `ValueError` if any required name is missing or an unexpected name
is supplied.  Raises `RuntimeError` on C ABI failure.
